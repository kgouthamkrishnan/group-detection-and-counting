from pathlib import Path
import random
import shutil

DATASET = Path("group_dataset")

TRAIN_IMG = DATASET / "images" / "train"
TRAIN_LBL = DATASET / "labels" / "train"

VAL_IMG = DATASET / "images" / "val"
VAL_LBL = DATASET / "labels" / "val"

TEST_IMG = DATASET / "images" / "test"
TEST_LBL = DATASET / "labels" / "test"

# Find images currently in train
images = sorted(TRAIN_IMG.glob("*.jpg"))

pairs = []

for image in images:
    label = TRAIN_LBL / f"{image.stem}.txt"

    if label.exists():
        pairs.append((image, label))
    else:
        print(f"WARNING: Missing label: {image.name}")

print(f"Image + label pairs found: {len(pairs)}")

if len(pairs) == 0:
    raise RuntimeError("No image-label pairs found.")

# Shuffle
random.seed(42)
random.shuffle(pairs)

total = len(pairs)

train_count = int(total * 0.80)
val_count = int(total * 0.10)

train_data = pairs[:train_count]
val_data = pairs[train_count:train_count + val_count]
test_data = pairs[train_count + val_count:]

print()
print("NEW SPLIT")
print("=" * 40)
print(f"Total : {total}")
print(f"Train : {len(train_data)}")
print(f"Val   : {len(val_data)}")
print(f"Test  : {len(test_data)}")


def move_pairs(data, image_dir, label_dir):

    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    for image, label in data:
        shutil.move(str(image), str(image_dir / image.name))
        shutil.move(str(label), str(label_dir / label.name))


print("\nMoving TRAIN...")
# Train files already belong in train, so leave them there.

print("Moving VAL...")
move_pairs(val_data, VAL_IMG, VAL_LBL)

print("Moving TEST...")
move_pairs(test_data, TEST_IMG, TEST_LBL)


print("\n" + "=" * 40)
print("DATASET SPLIT COMPLETED")
print("=" * 40)
print(f"Train : {len(list(TRAIN_IMG.glob('*.jpg')))}")
print(f"Val   : {len(list(VAL_IMG.glob('*.jpg')))}")
print(f"Test  : {len(list(TEST_IMG.glob('*.jpg')))}")
