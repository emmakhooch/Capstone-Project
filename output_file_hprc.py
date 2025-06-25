import os
import zipfile
import random
from pathlib import Path

# ----------------------------
# CONFIG
# ----------------------------
source_dir = "/scratch/user/e_hooch/food-101/food-101/food-101/images"
target_classes = ["apple_pie", "sushi", "spaghetti_bolognese"]
images_per_class = 10  # you can increase this if needed
output_zip = "/scratch/user/e_hooch/food_demo_sample.zip"

# ----------------------------
# ZIPPING
# ----------------------------
with zipfile.ZipFile(output_zip, "w") as zipf:
    for class_name in target_classes:
        class_path = os.path.join(source_dir, class_name)
        images = [f for f in os.listdir(class_path) if f.endswith(".jpg")]
        sampled = random.sample(images, min(images_per_class, len(images)))
        for img in sampled:
            full_img_path = os.path.join(class_path, img)
            arcname = os.path.join("images", class_name, img)  # relative path inside zip
            zipf.write(full_img_path, arcname=arcname)

print(f"✅ Done! Zip saved to: {output_zip}")