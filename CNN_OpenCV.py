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

train_txt = "/scratch/user/e_hooch/food-101/meta/train.txt"
test_txt = "/scratch/user/e_hooch/food-101/meta/test.txt"
image_root = "/scratch/user/e_hooch/food-101/images"
output_dir = "/scratch/user/e_hooch"
seed = 42

# -------------------------------
# FUNCTION: FAST SPLIT + FALSE LABELS
# -------------------------------
def split_image_list_with_false_labels(train_txt, test_txt, image_root, output_dir, seed=42):
    """Split dataset into 60/20/20 and generate false labels for the last 20%."""
    with open(train_txt, 'r') as f1, open(test_txt, 'r') as f2:
        image_lines = [line.strip() for line in f1.readlines() + f2.readlines()]

    total = len(image_lines)
    print(f"Total image entries: {total}")

    # Shuffle
    np.random.seed(seed)
    np.random.shuffle(image_lines)

    # Split 60/20/20
    i1 = int(0.6 * total)
    i2 = int(0.8 * total)
    train_split = image_lines[:i1]
    test_split = image_lines[i1:i2]
    fp_split = image_lines[i2:]

    # Save split files
    def save_list(lst, name):
        path = os.path.join(output_dir, name)
        with open(path, 'w') as f:
            for line in lst:
                f.write(f"{line}\n")
        print(f"Saved {len(lst)} entries to {path}")

    save_list(train_split, "train_split.txt")
    save_list(test_split, "test_split.txt")
    save_list(fp_split, "false_positives_split.txt")

    # Create false labels
    def create_false_labels(image_list, image_root):
        false_labels = []
        all_classes = sorted([d for d in os.listdir(image_root) if not d.startswith('.') and os.path.isdir(os.path.join(image_root, d))])
        for path in image_list:
            true_class = path.split('/')[0]
            candidates = [c for c in all_classes if c != true_class]
            false_class = random.choice(candidates)
            false_labels.append((path, false_class))
        return false_labels

    false_labels = create_false_labels(fp_split, image_root)

    # Save false labels
    fp_label_path = os.path.join(output_dir, "false_positives_labels.txt")
    with open(fp_label_path, 'w') as f:
        for original_path, false_label in false_labels:
            f.write(f"{original_path} -> {false_label}\n")
    print(f"Saved false positive labels to {fp_label_path}")

# -------------------------------
# EXECUTE
# -------------------------------
if __name__ == "__main__":
    split_image_list_with_false_labels(train_txt, test_txt, image_root, output_dir, seed=seed)

train_data_path = "/scratch/user/e_hooch/train_split.txt"
test_data_path = "/scratch/user/e_hooch/test_split.txt"
fp_data_path = "/scratch/user/e_hooch/false_positives_split.txt"
image_root = "/scratch/user/e_hooch/food-101/images"

# Set temporary resolution for faster processing
TARGET_RES = (288, 288)  # Half of 3024x4032 #for now to increase speed for demo

# Dataset class
for path in [train_data_path, test_data_path, image_root]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"ERROR: {path} not found!")
    
print("train_data_path =", train_data_path)
print("image_root =", image_root)

assert train_data_path is not None and os.path.exists(train_data_path), "train_data_path missing or invalid"
assert image_root is not None and os.path.exists(image_root), "image_root missing or invalid"
# Safe Dataset class

basic_transform = transforms.Compose([
    transforms.ToTensor(),  # Converts to (C, H, W) float tensor in [0,1]
])

class ImageDataset(Dataset):
    def __init__(self, txt_file, image_root, target_res=TARGET_RES):
        print(f"Starting ImageDataset init...")

        if not isinstance(txt_file, str) or not os.path.exists(txt_file):
            raise FileNotFoundError(f"ERROR: Data file {txt_file} not found!")
        if not isinstance(image_root, str) or not os.path.exists(image_root):
            raise FileNotFoundError(f"ERROR: Image root directory {image_root} not found!")

        with open(txt_file, 'r') as f:
            lines = f.readlines()
            print(f"Found {len(lines)} entries in txt file")

        # Debug print for first 10 lines
        print("\nPreviewing raw lines from split file:")
        for i in range(min(10, len(lines))):
            print(f"[{i}] {repr(lines[i])}")

        self.image_paths = [
            os.path.join(image_root, line.strip() + ".jpg")
            for line in lines if line.strip()
        ]

        print("\nSample final paths after .strip() + .jpg:")
        for i, p in enumerate(self.image_paths[:5]):
            print(f"[{i}] {repr(p)}")

        print(f"{len(self.image_paths)} valid images found after filtering.")

        if any(p is None for p in self.image_paths):
            raise ValueError("Found NoneType in self.image_paths!")

        if len(self.image_paths) == 0:
            raise ValueError("ERROR: No valid images found after filtering missing files!")

        self.target_res = target_res
        self.transform = basic_transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            
            img = Image.open(img_path).convert("RGB")
            img = img.resize(self.target_res[::-1])  # (width, height)
            img_tensor = self.transform(img)  # Convert to torch tensor

            
            return img_tensor

        except Exception as e:
            print(f"Skipping {img_path}: {e}")
            return torch.zeros((3, *self.target_res), dtype=torch.float32)

# Manual load check for a known image
known_image = "/scratch/user/e_hooch/food-101/images/apple_pie/1005649.jpg"
print("\nChecking if a known image exists:")
print(f"Does {known_image} exist?", os.path.exists(known_image))

try:
    img = Image.open(known_image).convert("RGB")
    img = img.resize((288, 288))
    img_np = np.array(img)
    print(f"Manually loaded {known_image}")
    print(f"Shape: {img_np.shape}, min: {img_np.min()}, max: {img_np.max()}")
except Exception as e:
    print(f"Error loading known image: {e}")

# Updated save_images_to_numpy without max() check

def save_images_to_numpy(dataset, output_file, batch_size=50, num_workers=8):
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, persistent_workers=True)

    total_images = len(dataset)
    image_shape = (*TARGET_RES, 3)

    print(f"Saving {total_images} images to memmap: {output_file}...")

    np_memmap = np.memmap(output_file, dtype='uint8', mode='w+', shape=(total_images, *image_shape))

    index = 0
    total_skipped = 0

    for batch in tqdm(dataloader, desc="Processing Batches"):
        clean_batch = []
        skipped = 0

        for img in batch:
            if isinstance(img, torch.Tensor):
                if img.numel() == 0 or img.sum().item() == 0:
                    skipped += 1
                    continue
                try:
                    np_img = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    if np_img.shape == image_shape:
                        clean_batch.append(np_img)
                    else:
                        skipped += 1
                except Exception as e:
                    print(f"Conversion failed: {e}")
                    skipped += 1
            else:
                skipped += 1

        if skipped > 0:
            print(f"Skipped {skipped} blank or invalid images in this batch")
            total_skipped += skipped

        if clean_batch:
            batch_arr = np.stack(clean_batch)
            np_memmap[index:index + len(batch_arr)] = batch_arr
            index += len(batch_arr)

    np_memmap.flush()
    del np_memmap  # Close the file
    del dataloader  # Cleanup

    print(f"Memmap saved at: {output_file}")
    print(f"Total valid images written: {index}")
    if total_skipped > 0:
        print(f"Total skipped images: {total_skipped}")
    if index == 0:
        print("No valid images written! Check dataset integrity or path resolution.")

    return output_file


# Load and inspect saved files
# --- Run the main function and capture important values ---
train_dataset = ImageDataset(train_data_path, image_root)
test_dataset = ImageDataset(test_data_path, image_root)
fp_dataset = ImageDataset(test_data_path, image_root)

save_images_to_numpy(train_dataset, "/scratch/user/e_hooch/train_images.memmap", batch_size=50, num_workers=8)
save_images_to_numpy(test_dataset, "/scratch/user/e_hooch/test_images.memmap", batch_size=50, num_workers=8)
save_images_to_numpy(fp_dataset, "/scratch/user/e_hooch/fp_images.memmap", batch_size=50, num_workers=8)

train_images = np.memmap("/scratch/user/e_hooch/train_images.memmap", dtype='uint8', mode='r', shape=(60600, 288, 288, 3))
print("Loaded memmap. Shape:", train_images.shape)

test_images = np.memmap("/scratch/user/e_hooch/test_images.memmap", dtype='uint8', mode='r', shape=(20200, 288, 288, 3))
print("Loaded memmap. Shape:", test_images.shape)

fp_images = np.memmap("/scratch/user/e_hooch/fp_images.memmap", dtype='uint8', mode='r', shape=(20200, 288, 288, 3))
print("Loaded memmap. Shape:", fp_images.shape)

# Add file existence test for a known image
print("\nChecking if a known image exists:")
known_image = "/scratch/user/e_hooch/food-101/images/apple_pie/1005649.jpg"
print(f"Does {known_image} exist?", os.path.exists(known_image))

# Debug raw memmap output
print("\nRaw memmap image stats:")
print("Train memmap shape:", train_images.shape)
print("First image pixel min/max:", train_images[0].min(), train_images[0].max())
# -------------------------------
# FUNCTION: DISPLAY RANDOM IMAGE
#Generate random images from Food-101 dataset
#import random

#classes = ['apple_pie', 'baby_back_ribs', 'baklava', 'beef_carpaccio', 'beef_tartare', 'beet_salad', 'beignets', 'bibimbap', 'bread_pudding', 'breakfast_burrito']
#def show_random_image(txt_file, image_root, target_res=(3024, 4032)):
    #if not os.path.exists(txt_file):
       # print(f"Error: {txt_file} does not exist!")
       # return

    #with open(txt_file, 'r') as file:
       # image_paths = file.readlines()

    #if not image_paths:
        #print("Error: No image paths found in the file.")
       # return

   # random_image_path = random.choice(image_paths).strip() + ".jpg"
   # full_image_path = os.path.join(image_root, random_image_path)

   # if os.path.exists(full_image_path):
       # image = cv2.imread(full_image_path)
       # image = cv2.resize(image, target_res)
       # cv2.imshow("Random Food Image", image)
       # cv2.waitKey(0)
       # cv2.destroyAllWindows()
    #else:
# ------------------ GENERATING BOUNDING BOXES ------------------
import json
from PIL import ImageDraw
# Setup
# Constants
from ultralytics import YOLO  # type: ignore

# ------------------------- CONFIG -------------------------
MEMMAP_SHAPE = (288, 288, 3)
BATCH_SIZE = 16
NUM_WORKERS = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONFIDENCE_THRESHOLD = 0.25
TARGET_RES = (288, 288)
NUM_CLASSES = 101
LABEL_SMOOTHING = 0.1
LR = 1e-4
NUM_EPOCHS = 50
USE_AMP = torch.cuda.is_available()

splits = {
    "train": {
        "memmap": "/scratch/user/e_hooch/train_images.memmap",
        "txt": "/scratch/user/e_hooch/train_split.txt"
    },
    "test": {
        "memmap": "/scratch/user/e_hooch/test_images.memmap",
        "txt": "/scratch/user/e_hooch/test_split.txt"
    },
    "fp": {
        "memmap": "/scratch/user/e_hooch/fp_images.memmap",
        "txt": "/scratch/user/e_hooch/false_positives_split.txt"
    },
}

save_dir = "/scratch/user/e_hooch/bboxes_memmap_yolo"
os.makedirs(save_dir, exist_ok=True)

# ------------------------- DATASET -------------------------
class MemmapImageDataset(Dataset):
    def __init__(self, memmap_path, txt_path):
        self.image_ids = self._load_txt(txt_path)
        self.num_images = len(self.image_ids)
        self.memmap = np.memmap(memmap_path, dtype='uint8', mode='r', shape=(self.num_images, *MEMMAP_SHAPE))

    def _load_txt(self, path):
        with open(path, 'r') as f:
            return [line.strip() for line in f if line.strip()]

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        arr = self.memmap[idx]
        return Image.fromarray(arr), self.image_ids[idx]

def yolo_collate(batch):
    images, ids = zip(*batch)
    return list(images), list(ids)

# ------------------------- YOLO INFERENCE -------------------------
def run_yolo_inference(split, memmap_path, txt_path, model):
    print(f"Starting YOLO detection on '{split}'...")

    dataset = MemmapImageDataset(memmap_path, txt_path)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, collate_fn=yolo_collate)

    bbox_dict = {}
    for images, ids in tqdm(loader, desc=f"Inferencing '{split}'", unit="batch"):
        results = model.predict(images, conf=CONFIDENCE_THRESHOLD, device=DEVICE, verbose=False)
        for r, img_id in zip(results, ids):
            boxes = []
            if r.boxes and len(r.boxes.xyxy):
                for b in r.boxes.xyxy.cpu().numpy():
                    x1, y1, x2, y2 = b[:4]
                    boxes.append([float(x1), float(y1), float(x2), float(y2)])
            bbox_dict[img_id] = boxes

    output_json = os.path.join(save_dir, f"{split}_bounding_boxes.json")
    with open(output_json, 'w') as f:
        json.dump(bbox_dict, f, indent=2)
    print(f"Saved {len(bbox_dict)} entries to {output_json}")

# ------------------------- MAIN -------------------------
if __name__ == "__main__":
    model_path = "/scratch/user/e_hooch/yolov5su.pt"
    model = YOLO(model_path)

    for split, paths in splits.items():
        run_yolo_inference(split, paths["memmap"], paths["txt"], model)

    print("\n All splits processed successfully.")

train_memmap_path = "/scratch/user/e_hooch/train_images.memmap"
test_memmap_path = "/scratch/user/e_hooch/test_images.memmap"
fp_memmap_path = "/scratch/user/e_hooch/fp_images.memmap"
bbox_json_path = "/scratch/user/e_hooch/bboxes_memmap_yolo/train_bounding_boxes.json"
weights_path = "/scratch/user/e_hooch/resnet50-weights.pth"

train_txt = "/scratch/user/e_hooch/train_split.txt"
test_txt = "/scratch/user/e_hooch/test_split.txt"
fp_txt = "/scratch/user/e_hooch/false_positives_split.txt"
fp_label_txt = "/scratch/user/e_hooch/false_positives_labels.txt"
# ------------------------- CONFIG -------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = torch.cuda.is_available()
TARGET_RES = (288, 288)
BATCH_SIZE = 32
NUM_EPOCHS = 50
LR = 1e-4
LABEL_SMOOTHING = 0.1
NUM_CLASSES = 101

memmap_root = "/scratch/user/e_hooch"
splits = ["train", "test", "fp"]
weights_path = os.path.join(memmap_root, "resnet50-weights.pth")
bbox_dir = os.path.join(memmap_root, "bboxes_memmap_yolo")

# ------------------------- UTILS -------------------------
def load_bounding_boxes(split):
    with open(os.path.join(bbox_dir, f"{split}_bounding_boxes.json"), 'r') as f:
        return json.load(f)

def load_split_txt(txt_path):
    with open(txt_path, 'r') as f:
        return [line.strip() for line in f.readlines() if line.strip()]

# ------------------------- TRANSFORM -------------------------
crop_transform = transforms.Compose([
    transforms.Resize(TARGET_RES),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# ------------------------- DATASET -------------------------
class CroppedObjectDataset(Dataset):
    def __init__(self, memmap_path, txt_path, bbox_json, class_to_idx, transform=None):
        self.image_ids = load_split_txt(txt_path)
        self.bbox_dict = bbox_json
        self.transform = transform
        self.class_to_idx = class_to_idx
        self.memmap = np.memmap(memmap_path, dtype='uint8', mode='r',
                                shape=(len(self.image_ids), *TARGET_RES, 3))
        self.samples = self._build_samples()

    def _build_samples(self):
        samples = []
        for idx, img_id in enumerate(self.image_ids):
            boxes = self.bbox_dict.get(img_id, [])
            label = img_id.split('/')[0]
            class_idx = self.class_to_idx[label]
            for box in boxes:
                if len(box) == 4:
                    samples.append((idx, box, class_idx))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_idx, box, label = self.samples[idx]
        img = self.memmap[img_idx].copy()
        img = Image.fromarray(img)
        x1, y1, x2, y2 = map(int, box)
        crop = img.crop((x1, y1, x2, y2))
        if self.transform:
            crop = self.transform(crop)
        return crop, label

# ------------------------- LOSS -------------------------
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        log_probs = F.log_softmax(pred, dim=1)
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (pred.size(1) - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        return torch.mean(torch.sum(-true_dist * log_probs, dim=1))

# ------------------------- MODEL -------------------------
from torchvision import models
from torch.cuda.amp import autocast, GradScaler
def build_model(num_classes):
    model = models.resnet50()
    state_dict = torch.load(weights_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

# ------------------------- MAIN -------------------------
if __name__ == "__main__":
    print(f"Running on {DEVICE}. AMP enabled: {USE_AMP}")

    train_bbox = load_bounding_boxes("train")
    train_txt = os.path.join(memmap_root, "train_split.txt")
    memmap_path = os.path.join(memmap_root, "train_images.memmap")

    with open(train_txt, 'r') as f:
        class_names = sorted({line.strip().split('/')[0] for line in f if line.strip()})
    class_to_idx = {cls: i for i, cls in enumerate(class_names)}

    dataset = CroppedObjectDataset(memmap_path, train_txt, train_bbox, class_to_idx, crop_transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

    model = build_model(NUM_CLASSES).to(DEVICE)
    criterion = LabelSmoothingCrossEntropy(smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    scaler = GradScaler() if USE_AMP else None

    # ------------------------- TRAIN -------------------------
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        for imgs, labels in tqdm(loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
            imgs, labels = imgs.to(DEVICE), torch.tensor(labels).to(DEVICE)
            optimizer.zero_grad()

            with autocast(enabled=USE_AMP):
                outputs = model(imgs)
                loss = criterion(outputs, labels)

            if USE_AMP:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()

        scheduler.step()
        print(f"Epoch {epoch+1} | Avg Loss: {total_loss / len(loader):.4f}")

    torch.save(model.state_dict(), os.path.join(memmap_root, "multi_item_model.pth"))
    print("Multi-item model saved.")


