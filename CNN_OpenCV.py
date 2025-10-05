import os
SCR = "/mnt/nfs-scratch/ECEN_403-404/nutrician_assistant"
os.environ.setdefault("ULTRALYTICS_CONFIG_DIR", f"{SCR}/.ultralytics_cfg")
os.environ.setdefault("ULTRALYTICS_CACHE_DIR",  f"{SCR}/.ultralytics_cache")
os.environ.setdefault("XDG_CONFIG_HOME",        f"{SCR}/.xdg_config")
os.environ.setdefault("XDG_CACHE_HOME",         f"{SCR}/.xdg_cache")
os.environ.setdefault("TORCH_HOME",             f"{SCR}/.torch")
os.environ.setdefault("TMPDIR",                  f"{SCR}/.tmp")
for d in (os.environ["ULTRALYTICS_CONFIG_DIR"], os.environ["ULTRALYTICS_CACHE_DIR"],
          os.environ["XDG_CONFIG_HOME"], os.environ["XDG_CACHE_HOME"],
          os.environ["TORCH_HOME"], os.environ["TMPDIR"]):
    os.makedirs(d, exist_ok=True)

from ultralytics import YOLO

#import needed packages and libraries
import numpy as np
import os
import random
import torch
import concurrent.futures
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image
from torch.quantization import QuantStub, DeQuantStub
import multiprocessing
from torch import nn
from PIL import Image
from ultralytics import YOLO
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
import json
from torchvision import models

train_txt = "/mnt/nfs-scratch/ECEN_403-404/nutrician_assistant/datasets/food-101/meta/train.txt"
test_txt = "/mnt/nfs-scratch/ECEN_403-404/nutrician_assistant/datasets/food-101/meta/test.txt"



# -------------------------------
# FUNCTION: FAST SPLIT + FALSE LABELS
# -------------------------------
# Edited version of CNN_OpenCV.py with shard-by-shard execution for
# splitting, saving memmap, running YOLO, training, and deleting each shard after training.


# ------------------------- CONFIG -------------------------
ROOT = "/mnt/nfs-scratch/ECEN_403-404/nutrician_assistant"
IMG_ROOT = os.path.join(ROOT, "datasets/food-101/images")
BATCH_SIZE = 32
NUM_WORKERS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONFIDENCE_THRESHOLD = 0.25
NUM_CLASSES = 102
NON_FOOD_LABEL = "non_food"
LABEL_SMOOTHING = 0.1
LR = 1e-4
NUM_EPOCHS = 50
USE_AMP = torch.cuda.is_available()
SHARD_SIZE = 100
TARGET_RES = (288, 288)
ORIG_RES = (1512, 2016)
MEMMAP_SHAPE = (*ORIG_RES, 3)
weights_path = os.path.join(ROOT, "weights/resnet50-weights.pth")
bbox_dir = os.path.join(ROOT, "bboxes")
os.makedirs(bbox_dir, exist_ok=True)
model_path = os.path.join(ROOT, "weights/yolo11n.pt")
yolo_model = YOLO(model_path)

crop_transform = transforms.Compose([
    transforms.Resize(TARGET_RES),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# ------------------------- SPLIT FUNCTION -------------------------
def split_image_list_with_false_labels(train_txt, test_txt, image_root, output_dir, seed=42):
    with open(train_txt, 'r') as f1, open(test_txt, 'r') as f2:
        image_lines = [line.strip() for line in f1.readlines() + f2.readlines() if line.strip()]

    total = len(image_lines)
    print(f"Total image entries: {total}")

    np.random.seed(seed)
    np.random.shuffle(image_lines)

    i1 = int(0.6 * total)
    i2 = int(0.8 * total)
    train_split = image_lines[:i1]
    test_split = image_lines[i1:i2]
    fp_split = image_lines[i2:]

    def save_list(lst, filename):
        path = os.path.join(output_dir, filename)
        with open(path, 'w') as f:
            for line in lst:
                f.write(f"{line}\n")
        print(f"Saved {len(lst)} entries to {path}")

    save_list(train_split, "train_split.txt")
    save_list(test_split, "test_split.txt")
    save_list(fp_split, "false_positives_split.txt")

    def create_false_labels(image_list, image_root):
        false_labels = []
        all_classes = sorted([
            d for d in os.listdir(image_root)
            if os.path.isdir(os.path.join(image_root, d)) and not d.startswith(".")
        ])
        for path in image_list:
            true_class = path.split('/')[0]
            candidates = [cls for cls in all_classes if cls != true_class]
            false_class = random.choice(candidates)
            false_labels.append((path, false_class))
        return false_labels

    false_labels = create_false_labels(fp_split, image_root)

    fp_label_path = os.path.join(output_dir, "false_positives_labels.txt")
    with open(fp_label_path, 'w') as f:
        for original_path, false_label in false_labels:
            f.write(f"{original_path} -> {false_label}\n")
    print(f"Saved {len(false_labels)} false labels to {fp_label_path}")

def is_valid_box(box, img_w, img_h, min_area_ratio=0.01, max_area_ratio=0.7,
                        min_aspect_ratio=0.3, max_aspect_ratio=3.0
                                                                ):
    """
    Check if a YOLO box is valid food candidate or likely non-food.

    Args:
        box (list): [x1, y1, x2, y2] in pixels
        img_w, img_h (int): image dimensions
        min_area_ratio (float): lower bound of box area relative to image
        max_area_ratio (float): upper bound of box area relative to image
        min_aspect_ratio (float): lower bound of aspect ratio (w/h or h/w)
        max_aspect_ratio (float): upper bound of aspect ratio
    """
    x1, y1, x2, y2 = map(int, box)
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return False

    area = w * h
    img_area = img_w * img_h
    aspect_ratio = max(w / h, h / w)

    # Area-based filtering
    if area < min_area_ratio * img_area:  # too small
        return False
    if area > max_area_ratio * img_area:  # too large (plate/full table)
        return False

    # Aspect ratio filtering
    if aspect_ratio < min_aspect_ratio or aspect_ratio > max_aspect_ratio:
        return False

    return True

    
# ------------------------------ DATASET CLASSES ------------------------------
class MemmapBboxDataset(Dataset):
    def __init__(self, memmap_path, image_ids, bboxes, transform=None):
        self.memmap = np.memmap(memmap_path, dtype='uint8', mode='r',
                                shape=(len(image_ids), *MEMMAP_SHAPE))
        self.image_ids = image_ids
        self.bboxes = bboxes
        self.transform = transform
        self.samples = self._build_samples()

    def _build_samples(self):
        samples = []
        for idx, img_id in enumerate(self.image_ids):
            true_class = img_id.split('/')[0]  # folder = Food-101 label
            boxes = self.bboxes.get(img_id, [])
            if not boxes:
                continue

            img_h, img_w = MEMMAP_SHAPE[:2]
            valid_boxes = [b for b in boxes if is_valid_box(b, img_w, img_h)]

            for j, box in enumerate(valid_boxes):
                if j == 0:
                    label = true_class
                else:
                    label = NON_FOOD_LABEL
                samples.append((idx, box, label))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_idx, box, label = self.samples[idx]
        img = Image.fromarray(self.memmap[img_idx])
        crop = img.crop(tuple(map(int, box)))
        if self.transform:
            crop = self.transform(crop)
        return crop, class_to_idx[label]

# -----------------------

# ------------------------- CONFIG -------------------------
# Constants

all_classes = sorted([
    d for d in os.listdir(IMG_ROOT)
    if os.path.isdir(os.path.join(IMG_ROOT, d)) and not d.startswith(".")
])
all_classes.append(NON_FOOD_LABEL)  # add at the end
class_to_idx = {cls: i for i, cls in enumerate(all_classes)}

# ------------------------- UTILS -------------------------
def load_split_txt(txt_path):
    with open(txt_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def save_json(obj, path):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)
# ------------------------- STRATIFIED SPLITTING -------------------------
from collections import defaultdict
def stratified_shards(all_ids, shard_size):
    """Split IDs into balanced shards across classes."""
    per_class = defaultdict(list)
    for iid in all_ids:
        cls = iid.split('/')[0]
        per_class[cls].append(iid)

    # shuffle each class
    for cls in per_class:
        random.shuffle(per_class[cls])

    shards = []
    while any(per_class.values()):
        shard = []
        for cls, items in list(per_class.items()):
            take = min(len(items), max(1, shard_size // len(per_class)))
            shard.extend(items[:take])
            per_class[cls] = items[take:]
        if shard:
            random.shuffle(shard)
            shards.append(shard)
    return shards

# ------------------------- IMAGE DATASET -------------------------
class HighResImageDataset(Dataset):
    def __init__(self, data_source, image_root, target_res=ORIG_RES):
        """
        Args:
            data_source (str or list): Either a path to a .txt file with image IDs
                                       or a Python list of image IDs.
            image_root (str): Root directory containing Food-101 images.
            target_res (tuple): (H, W) resolution to resize images to.
        """
        # If passed a txt file, load IDs from it
        if isinstance(data_source, str) and data_source.endswith(".txt"):
            with open(data_source, "r") as f:
                self.image_ids = [line.strip() for line in f if line.strip()]
        # If passed a list, use it directly
        elif isinstance(data_source, (list, tuple)):
            self.image_ids = list(data_source)
        else:
            raise ValueError("data_source must be a .txt file path or a list of image IDs")

        self.image_paths = [os.path.join(image_root, line + ".jpg") for line in self.image_ids]
        self.target_res = target_res

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        try:
            with Image.open(path) as img:
                img = img.convert("RGB")
                img = img.resize(self.target_res[::-1])  # PIL expects (W, H)
                arr = np.array(img, dtype=np.uint8)
                if arr.shape != (*self.target_res, 3):
                    raise ValueError(f"Invalid shape {arr.shape} for image: {path}")
                return arr
        except Exception as e:
            print(f"⚠️ Skipping {path}: {e}")
            return np.zeros((*self.target_res, 3), dtype=np.uint8)


# ------------------------- MEMMAP SAVING -------------------------
def save_memmap_shard(dataset, image_ids, shard_id, output_prefix, output_dir=None):
    """
    Save a memmap shard and also create a matching shard_{id}_ids.txt file.

    Args:
        dataset: A Dataset object that yields (H, W, 3) np.uint8 arrays.
        image_ids (list): List of image IDs corresponding to this shard.
        shard_id (int): Shard index number.
        output_prefix (str): Prefix for memmap file path (before `_shard{}`).
        output_dir (str, optional): Directory for saving the .txt file.
                                    Defaults to same directory as output_prefix.

    Returns:
        str: Path to the saved memmap shard.
        str: Path to the saved IDs text file.
    """
    shard_len = len(dataset)
    memmap_path = f"{output_prefix}_shard{shard_id}.memmap"
    memmap = np.memmap(memmap_path, dtype='uint8', mode='w+', shape=(shard_len, *ORIG_RES, 3))

    for i in tqdm(range(shard_len), desc=f"Shard {shard_id}"):
        memmap[i] = dataset[i]

    memmap.flush()
    del memmap

    # Figure out where to save shard IDs file
    if output_dir is None:
        output_dir = os.path.dirname(output_prefix)
    os.makedirs(output_dir, exist_ok=True)

    ids_path = os.path.join(output_dir, f"shard_{shard_id}_ids.txt")
    with open(ids_path, "w") as f:
        for img_id in image_ids:
            f.write(f"{img_id}\n")

    print(f"[Shard {shard_id}] Saved memmap: {memmap_path}")
    print(f"[Shard {shard_id}] Saved IDs file: {ids_path}")

    return memmap_path, ids_path

# ================================
# SHARD CREATION + SAVING
# ================================

def create_memmap_shards(split_txt, image_root, output_dir, prefix, shard_size=1000):
    """
    Create memmap shards from a split text file and also save matching shard_{id}_ids.txt files.

    Args:
        split_txt (str): Path to .txt file listing image IDs (class/image_id without .jpg).
        image_root (str): Path to root images folder (Food-101/images).
        output_dir (str): Directory where shards + ids files will be saved.
        prefix (str): Prefix for shard file names (e.g., 'train', 'test', 'fp').
        shard_size (int): Number of images per shard.

    Returns:
        list of (memmap_path, ids_path): Paths to created memmap and IDs files.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load all image IDs
    with open(split_txt, "r") as f:
        all_ids = [line.strip() for line in f if line.strip()]

    print(f"Loaded {len(all_ids)} IDs from {split_txt}")

    # Split into shards
    shards = [all_ids[i:i+shard_size] for i in range(0, len(all_ids), shard_size)]
    created = []

    for shard_id, shard_ids in enumerate(shards):
        print(f"\n=== Building shard {shard_id} ({len(shard_ids)} images) ===")

        # Build dataset for this shard
        dataset = HighResImageDataset(
            data_source=shard_ids,
            image_root=image_root,
            target_res=ORIG_RES
        )

        # Save memmap + IDs file
        memmap_prefix = os.path.join(output_dir, f"{prefix}")
        memmap_path, ids_path = save_memmap_shard(dataset, shard_ids, shard_id, memmap_prefix, output_dir)

        created.append((memmap_path, ids_path))

    return created

# ------------------------- YOLO INFERENCE -------------------------
'''
class MemmapDatasetForYOLO(Dataset):
    def __init__(self, memmap_path, image_ids):
        self.image_ids = image_ids
        self.memmap = np.memmap(
            memmap_path, dtype='uint8', mode='r',
            shape=(len(image_ids), *ORIG_RES, 3)
        )

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        arr = np.array(self.memmap[idx], dtype=np.uint8)   # force copy
        arr = np.ascontiguousarray(arr)                    # ensure contiguous memory
        return arr, self.image_ids[idx]                    # <-- return ndarray, not PIL
'''
from PIL import Image

def resolve_img_path(img_id: str) -> str:
    iid = img_id.strip().rstrip("\r\n")
    return os.path.join(IMG_ROOT, f"{iid}.jpg")

DATASET_ROOT = "/mnt/nfs-scratch/ECEN_403-404/nutrician_assistant/datasets/food-101"

# ------------------------- YOLO LOADING -------------------------
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import os, json
from tqdm import tqdm

# Load YOLO11n model
model_path = os.path.join(ROOT, "weights/yolo11n.pt")
yolo_model = YOLO(model_path)

# ------------------------- YOLO INFERENCE -------------------------

def run_yolo_on_shard(
    shard_ids_file, 
    shard_id, 
    yolo_model, 
    dataset_root, 
    output_dir, 
    yolo_size=640,
    log_every=100
):
    """
    Run YOLO on a shard of images, pre-resizing each image to YOLO's expected input size.

    Args:
        shard_ids_file (str): Path to .txt file with relative image paths (class/image_id).
        shard_id (int): Shard index for logging.
        yolo_model: Loaded YOLO model (ultralytics).
        dataset_root (str): Root of dataset containing 'images'.
        output_dir (str): Directory to save JSON bounding box results.
        yolo_size (int): Image resize size for YOLO (default=640).
        log_every (int): How often to print debug info.
    """
    print(f"\n=== Running YOLO on shard {shard_id} ===")

    # Load shard image IDs
    with open(shard_ids_file, "r") as f:
        shard_ids = [line.strip() for line in f if line.strip()]

    bbox_dict = {}
    failed_count, success_count, skipped_count = 0, 0, 0

    for idx, rel_path in enumerate(tqdm(shard_ids, desc=f"Shard {shard_id}")):
        # Build absolute path
        full_path = os.path.join(dataset_root, "images", rel_path)
        if not full_path.lower().endswith(".jpg"):
            full_path += ".jpg"

        # Periodic debug message
        if idx % log_every == 0:
            print(f"[DEBUG] Processing {idx}/{len(shard_ids)}: {rel_path}")

        # Load with OpenCV, fallback to PIL
        test_img = cv2.imread(full_path)
        if test_img is None:
            try:
                pil_img = Image.open(full_path).convert("RGB")
                test_img = np.array(pil_img, dtype=np.uint8)
            except Exception as e:
                if idx % log_every == 0:
                    print(f"[YOLO] Failed to load: {full_path}, error={e}")
                failed_count += 1
                continue

        # Validate image type
        if not isinstance(test_img, np.ndarray):
            if idx % log_every == 0:
                print(f"[YOLO] Invalid image object: {full_path}, got {type(test_img)}")
            failed_count += 1
            continue

        # Normalize + ensure 3 channels
        if test_img.dtype != np.uint8:
            test_img = test_img.astype(np.uint8)
        if len(test_img.shape) == 2:
            test_img = cv2.cvtColor(test_img, cv2.COLOR_GRAY2RGB)
        elif test_img.shape[2] != 3:
            test_img = test_img[:, :, :3]

        # Pre-resize for YOLO
        test_img = cv2.resize(test_img, (yolo_size, yolo_size), interpolation=cv2.INTER_LINEAR)

        # Run YOLO
        try:
            results = yolo_model.predict(source=test_img, conf=0.25, verbose=False)

            bboxes = []
            for r in results:
                for box in r.boxes:
                    b = box.xyxy[0].tolist()
                    cls = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    bboxes.append({"bbox": b, "class": cls, "conf": conf})

            bbox_dict[rel_path] = bboxes
            success_count += 1

        except Exception as e:
            if idx % log_every == 0:
                print(f"[YOLO] Prediction failed on {full_path}, error={e}")
            failed_count += 1
            continue

    # Final shard summary
    print(f"\n[Shard {shard_id}] Processed {len(shard_ids)} images")
    print(f"  Successful detections: {success_count}")
    print(f"  Failed/invalid: {failed_count}")

    # Save JSON output
    os.makedirs(output_dir, exist_ok=True)
    out_json = os.path.join(output_dir, f"train_bounding_boxes_shard{shard_id}.json")
    with open(out_json, "w") as f:
        json.dump(bbox_dict, f)

    print(f"[Shard {shard_id}] Saved {len(bbox_dict)} entries → {out_json}")

    return bbox_dict

# ------------------------------ CROPPED OBJECT DATASET ------------------------------
class CroppedObjectDataset(Dataset):
    def __init__(self, memmap_path, image_ids, bbox_dict, class_to_idx, transform=None, use_nonfood=False):
        self.memmap = np.memmap(memmap_path, dtype='uint8', mode='r',
                                shape=(len(image_ids), *MEMMAP_SHAPE))
        self.image_ids = image_ids
        self.bbox_dict = bbox_dict
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.use_nonfood = use_nonfood
        self.samples = self._build_samples()

    def _build_samples(self):
        samples = []
        img_h, img_w = ORIG_RES
        n_food, n_nonfood = 0, 0

        for i, img_id in enumerate(self.image_ids):
            label_name = img_id.split('/')[0]
            true_idx = self.class_to_idx.get(label_name)
            nonfood_idx = self.class_to_idx[NON_FOOD_LABEL]

            boxes = self.bbox_dict.get(img_id, [])
            if not boxes:
                continue

            # Validate boxes
            valid_boxes = [b for b in boxes if is_valid_box(b["bbox"], img_w, img_h)]
            rejected = [b for b in boxes if not is_valid_box(b["bbox"], img_w, img_h)]

            if valid_boxes and true_idx is not None:
                # First box → true food
                samples.append((i, valid_boxes[0]["bbox"], true_idx))
                n_food += 1

                # Other valid boxes → non_food (only if enabled)
                if self.use_nonfood:
                    for box in valid_boxes[1:]:
                        samples.append((i, box["bbox"], nonfood_idx))
                        n_nonfood += 1

            # Explicit rejected boxes always go to non_food if enabled
            if self.use_nonfood:
                for box in rejected:
                    samples.append((i, box["bbox"], nonfood_idx))
                    n_nonfood += 1

        total = len(samples)
        ratio = n_nonfood / max(n_food + n_nonfood, 1)
        print(f"[CroppedObjectDataset] Built {total} samples "
              f"(food={n_food}, non_food={n_nonfood}, ratio={ratio:.2f}, use_nonfood={self.use_nonfood})")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_idx, box, label = self.samples[idx]
        arr = self.memmap[img_idx]
        img = Image.fromarray(arr)
        crop = img.crop(tuple(map(int, box)))
        if self.transform:
            crop = self.transform(crop)
        return crop, label

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        log_probs = torch.nn.functional.log_softmax(pred, dim=1)
        true_dist = torch.zeros_like(log_probs)
        true_dist.fill_(self.smoothing / (pred.size(1) - 1))
        true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        return torch.mean(torch.sum(-true_dist * log_probs, dim=1))

# ================================
# TRAIN CLASSIFIER (Option 1 Sequential Fine-Tuning)
# ================================
def train_classifier(memmap_path, image_ids, bbox_dict, shard_id, prev_model_path=None, use_nonfood=False):
    global class_to_idx

    transform = transforms.Compose([
        transforms.Resize(TARGET_RES),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    dataset = CroppedObjectDataset(
        memmap_path, image_ids, bbox_dict, class_to_idx, transform, use_nonfood=use_nonfood
    )

    if len(dataset) == 0:
        print(f"[Shard {shard_id}] No training samples. Skipping shard.")
        return None

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=NUM_WORKERS, persistent_workers=True, pin_memory=True)

    # Load ResNet and expand head
    model = models.resnet50()
    checkpoint = torch.load(weights_path, map_location="cpu")
    model.load_state_dict({k: v for k, v in checkpoint.items() if not k.startswith("fc")}, strict=False)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model.to(DEVICE)

    # Optionally load previous shard weights for cumulative training
    if prev_model_path and os.path.exists(prev_model_path):
        print(f"[Shard {shard_id}] 🔄 Loading weights from {prev_model_path}")
        model.load_state_dict(torch.load(prev_model_path, map_location=DEVICE))

    # Warmup smoothing
    criterion = LabelSmoothingCrossEntropy(smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    scaler = GradScaler() if USE_AMP else None

    model.train()
    for epoch in range(NUM_EPOCHS):
        total_loss, total_correct, total_samples = 0.0, 0, 0

        # Optionally disable smoothing after 10 epochs
        if epoch == 10:
            criterion = nn.CrossEntropyLoss()
            print(f"[Shard {shard_id}] 🔄 Switched to plain CrossEntropyLoss (no smoothing)")

        for imgs, labels in tqdm(loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [shard {shard_id}]"):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            with autocast(enabled=USE_AMP):
                outputs = model(imgs)
                loss = criterion(outputs, labels)

            # Backprop
            if USE_AMP:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * imgs.size(0)

            # Accuracy
            _, preds = torch.max(outputs, 1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

        scheduler.step()

        avg_loss = total_loss / total_samples
        acc = total_correct / total_samples
        print(f"[Shard {shard_id}] Epoch {epoch+1} Loss: {avg_loss:.4f}, Accuracy: {acc*100:.2f}%")

    # Save checkpoint for chaining
    out_path = os.path.join(ROOT, f"multi_item_model_shard_{shard_id}.pth")
    torch.save(model.state_dict(), out_path)
    print(f"[Shard {shard_id}] ✅ Trained model saved to {out_path}")

    return out_path

# ================================
# MAIN SHARD-BY-SHARD PIPELINE (YOLO11n, Stratified, 2-Stage Training)
# ================================
import os

print("🚀 Starting stratified shard-by-shard training with YOLO11n...")

ROOT = "/mnt/nfs-scratch/ECEN_403-404/nutrician_assistant"
DATASET_ROOT = os.path.join(ROOT, "datasets/food-101")
IMG_ROOT = os.path.join(DATASET_ROOT, "images")
bbox_dir = os.path.join(ROOT, "bboxes")
shard_dir = os.path.join(ROOT, "shards")
os.makedirs(bbox_dir, exist_ok=True)
os.makedirs(shard_dir, exist_ok=True)

train_split_path = os.path.join(DATASET_ROOT, "train_split.txt")
if not os.path.exists(train_split_path):
    raise FileNotFoundError(f"❌ Missing train_split.txt at {train_split_path}")

# Read all training IDs
with open(train_split_path, "r") as f:
    all_train_ids = [line.strip() for line in f if line.strip()]
print(f"✅ Total training images: {len(all_train_ids)}")

# Make stratified shards
SHARD_SIZE = 1000
shards = stratified_shards(all_train_ids, SHARD_SIZE)
print(f"✅ Created {len(shards)} stratified shards")

# Load YOLO11n
from ultralytics import YOLO
yolo_weights = os.path.join(ROOT, "weights/yolo11n.pt")
if not os.path.exists(yolo_weights):
    raise FileNotFoundError(f"❌ Missing YOLO11n weights at {yolo_weights}")
yolo_model = YOLO(yolo_weights)
print(f"✅ Loaded YOLO model from {yolo_weights}")

# ====================
# Stage 1: Food-only training
# ====================
print("\n=== 🥗 Stage 1: Training food-only (no non_food) ===")
for shard_id, shard_ids in enumerate(shards):
    print(f"\n--- Processing shard {shard_id}/{len(shards)-1} ---")

    shard_ids_file = os.path.join(shard_dir, f"shard_{shard_id}_ids.txt")
    with open(shard_ids_file, "w") as f:
        for img_id in shard_ids:
            f.write(f"{img_id}\n")

    # Build memmap
    dataset = HighResImageDataset(shard_ids, IMG_ROOT, target_res=ORIG_RES)
    memmap_prefix = os.path.join(shard_dir, f"shard_{shard_id}")
    memmap_path, ids_path = save_memmap_shard(dataset, shard_ids, shard_id, memmap_prefix, shard_dir)

    # Run YOLO detection
    bbox_dict = run_yolo_on_shard(
        shard_ids_file=shard_ids_file,
        shard_id=shard_id,
        yolo_model=yolo_model,
        dataset_root=DATASET_ROOT,
        output_dir=bbox_dir,
        yolo_size=640
    )

    # Train classifier food-only
    train_classifier(
        memmap_path=memmap_path,
        image_ids=shard_ids,
        bbox_dict=bbox_dict,
        shard_id=shard_id,
        use_nonfood=False   # 🚫 skip non_food in stage 1
    )

    # Cleanup memmap
    if os.path.exists(memmap_path):
        os.remove(memmap_path)
        print(f"[Shard {shard_id}] 🗑️ Deleted memmap")

# ====================
# Stage 2: Fine-tune with non_food
# ====================
print("\n=== 🍔 Stage 2: Fine-tuning with non_food included ===")
for shard_id, shard_ids in enumerate(shards):
    shard_ids_file = os.path.join(shard_dir, f"shard_{shard_id}_ids.txt")
    memmap_prefix = os.path.join(shard_dir, f"shard_{shard_id}")
    dataset = HighResImageDataset(shard_ids, IMG_ROOT, target_res=ORIG_RES)
    memmap_path, ids_path = save_memmap_shard(dataset, shard_ids, shard_id, memmap_prefix, shard_dir)

    bbox_dict_path = os.path.join(bbox_dir, f"train_bounding_boxes_shard{shard_id}.json")
    if os.path.exists(bbox_dict_path):
        with open(bbox_dict_path, "r") as f:
            bbox_dict = json.load(f)
    else:
        print(f"⚠️ Missing bbox JSON for shard {shard_id}, skipping")
        continue

    # Fine-tune classifier with non_food included
    train_classifier(
        memmap_path=memmap_path,
        image_ids=shard_ids,
        bbox_dict=bbox_dict,
        shard_id=shard_id,
        use_nonfood=True   # ✅ include non_food in stage 2
    )

    if os.path.exists(memmap_path):
        os.remove(memmap_path)
        print(f"[Shard {shard_id}] 🗑️ Deleted memmap")

print("\n✅ Training complete: Stage 1 (food only) + Stage 2 (fine-tune with non_food)")



