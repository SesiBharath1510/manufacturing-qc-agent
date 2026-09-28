import os
import random
import cv2
import numpy as np

# 1. Setup dataset directories
dataset_root = os.path.abspath(os.path.join("data", "defect_dataset"))
train_img_dir = os.path.join(dataset_root, "images", "train")
val_img_dir = os.path.join(dataset_root, "images", "val")
train_lbl_dir = os.path.join(dataset_root, "labels", "train")
val_lbl_dir = os.path.join(dataset_root, "labels", "val")

for folder in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
    os.makedirs(folder, exist_ok=True)

# 2. Defect Classes
# 0 = Scratch (linear flaw), 1 = Void (surface pitting/hole), 2 = Crack (irregular fracture)
CLASSES = ["scratch", "void", "crack"]

def generate_synthetic_defect_sample(sample_idx, split):
    # Industrial metal plate background (brushed steel texture with slight noise)
    img_size = 320
    img = np.full((img_size, img_size, 3), 140, dtype=np.uint8)
    noise = np.random.randint(-15, 15, (img_size, img_size, 3), dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Pick a random defect class
    class_id = random.randint(0, 2)
    labels = []

    if class_id == 0:  # Scratch: high-contrast angled thin line
        x1, y1 = random.randint(40, 160), random.randint(40, 160)
        x2, y2 = x1 + random.randint(40, 120), y1 + random.randint(20, 80)
        cv2.line(img, (x1, y1), (x2, y2), (40, 40, 40), thickness=2)
        xmin, xmax = min(x1, x2) - 4, max(x1, x2) + 4
        ymin, ymax = min(y1, y2) - 4, max(y1, y2) + 4

    elif class_id == 1:  # Void: dark circular pit or crater
        cx, cy = random.randint(80, 240), random.randint(80, 240)
        radius = random.randint(12, 28)
        cv2.circle(img, (cx, cy), radius, (30, 30, 30), -1)
        cv2.circle(img, (cx, cy), radius + 2, (80, 80, 80), 1)
        xmin, xmax = cx - radius - 2, cx + radius + 2
        ymin, ymax = cy - radius - 2, cy + radius + 2

    else:  # Crack: zig-zag jagged line
        curr_x, curr_y = random.randint(60, 200), random.randint(60, 200)
        pts = [(curr_x, curr_y)]
        for _ in range(4):
            curr_x += random.randint(-15, 25)
            curr_y += random.randint(15, 30)
            pts.append((curr_x, curr_y))
        for i in range(len(pts) - 1):
            cv2.line(img, pts[i], pts[i + 1], (20, 20, 20), thickness=2)
        pts_np = np.array(pts)
        xmin, xmax = np.min(pts_np[:, 0]) - 5, np.max(pts_np[:, 0]) + 5
        ymin, ymax = np.min(pts_np[:, 1]) - 5, np.max(pts_np[:, 1]) + 5

    # Clamp coordinates to image boundaries
    xmin, xmax = max(0, xmin), min(img_size, xmax)
    ymin, ymax = max(0, ymin), min(img_size, ymax)

    # Normalize coordinates to [0.0, 1.0] for YOLO format
    x_center = ((xmin + xmax) / 2.0) / img_size
    y_center = ((ymin + ymax) / 2.0) / img_size
    width = (xmax - xmin) / float(img_size)
    height = (ymax - ymin) / float(img_size)

    labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

    # Save image and label
    filename = f"sample_{split}_{sample_idx:03d}"
    img_dest = os.path.join(train_img_dir if split == "train" else val_img_dir, f"{filename}.jpg")
    lbl_dest = os.path.join(train_lbl_dir if split == "train" else val_lbl_dir, f"{filename}.txt")

    cv2.imwrite(img_dest, img)
    with open(lbl_dest, "w") as f:
        f.writelines(labels)

# 3. Generate 30 training samples and 10 validation samples
print("⏳ Generating synthetic manufacturing defect dataset...")
for i in range(30):
    generate_synthetic_defect_sample(i, split="train")
for j in range(10):
    generate_synthetic_defect_sample(j, split="val")

# 4. Write YOLO dataset YAML file
yaml_content = f"""path: {dataset_root.replace(os.sep, '/')}
train: images/train
val: images/val

names:
  0: scratch
  1: void
  2: crack
"""

yaml_path = os.path.join(dataset_root, "defects.yaml")
with open(yaml_path, "w") as f:
    f.write(yaml_content)

print(f"✅ Defect dataset prepared successfully at:\n   {dataset_root}")
print(f"✅ Configuration YAML created at:\n   {yaml_path}")