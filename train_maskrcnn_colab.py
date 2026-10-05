# %% [markdown]
# # Train your own particle detector
#
# This notebook trains a Mask R-CNN model on **your** particle images, so TEMseg can segment
# your specific particles instead of its built-in model.
#
# **What you need**
# - A ZIP file with your data. Either:
#   - a **"COCO" or "Instances" export from the app** (one ZIP per session, or several session
#     ZIPs packed into one big ZIP), or
#   - any COCO-style dataset: images plus an annotations JSON with polygons.
# - Roughly: **50+ labelled images** gives a usable model, more is better. Fewer than ~20 will
#   mostly memorise the training images.
#
# **What happens here**
# 1. You run the cells top to bottom (click a cell, press Shift+Enter).
# 2. The notebook trains the model and shows you how well it did.
# 3. You download one file, `maskrcnn_best.pth`, and drop it into the app.
#
# **Time:** about 1 minute per image in your dataset with the free Colab GPU.
# Total for 100 images: around 1.5–2 hours. Colab will disconnect if the tab sleeps —
# keep it open, or pay for Colab Pro to avoid interruptions.
#
# **Before anything else:** in the Colab menu go to **Runtime → Change runtime type**
# and pick **T4 GPU** as the hardware accelerator. Without that, training is 10–50× slower.

# %%
!nvidia-smi -L
import torch

assert torch.cuda.is_available(), (
    "No GPU detected. Go to Runtime > Change runtime type and select T4 GPU, then rerun this cell."
)
print(f"Ready. GPU: {torch.cuda.get_device_name(0)}")

# %% [markdown]
# # Step 1 — Upload your data
#
# Run the cell below, then choose your data ZIP when the file picker appears.
# You can select several ZIPs at once.
#
# The app's exports work as-is:
# - **COCO export** — `original_image.png` + `annotations.coco.json`
# - **Instances export** — `original_image.png` + `instances.json`
#
# Exports that only contain statistics (no image, no annotations) are reported and skipped.

# %%
!pip install -q torchvision

import shutil
import zipfile
from pathlib import Path
from google.colab import files

DATA_DIR = Path("data")
if DATA_DIR.exists():
    shutil.rmtree(DATA_DIR)
DATA_DIR.mkdir()

uploaded = files.upload()  # pick your data ZIP(s)
for name in uploaded:
    zpath = Path(name)
    target = DATA_DIR / zpath.stem
    target.mkdir()
    with zipfile.ZipFile(zpath) as zf:
        zf.extractall(target)
    # session ZIPs nested inside the big ZIP
    for inner in target.rglob("*.zip"):
        with zipfile.ZipFile(inner) as zf:
            zf.extractall(inner.parent / inner.stem)
        inner.unlink()
    zpath.unlink()

print("Unpacked into", DATA_DIR)

# %%
import json
from collections import defaultdict

import cv2
import numpy as np


def polygons_from_segmentation(seg):
    if not isinstance(seg, list) or not seg:
        raise ValueError("unsupported segmentation (only polygons are supported)")
    out = []
    for part in seg:
        if isinstance(part, dict):
            raise ValueError(
                "This annotation uses compressed masks instead of outlines. "
                "Re-export your data from the app as a COCO export, or convert it to COCO polygons."
            )
        pts = np.asarray(part, dtype=np.float64).reshape(-1, 2)
        if len(pts) >= 3:
            out.append(pts)
    return out


def load_coco(json_path):
    with open(json_path) as f:
        coco = json.load(f)
    images = {im["id"]: im for im in coco.get("images", [])}
    anns = defaultdict(list)
    for ann in coco.get("annotations", []):
        if ann.get("iscrowd", 0):
            continue
        try:
            anns[ann["image_id"]].append(polygons_from_segmentation(ann["segmentation"]))
        except ValueError as e:
            print(f"Skipped one annotation in {json_path.name}: {e}")
    root = json_path.parent
    records = []
    for im in images.values():
        img_path = next((p for p in root.rglob(Path(im["file_name"]).name)), None)
        if img_path is not None:
            records.append({"image": img_path, "polygons": anns.get(im["id"], [])})
    return records


def load_instances(json_path):
    img_path = next(iter(json_path.parent.glob("original_image.png")), None)
    if img_path is None:
        img_path = next(iter(json_path.parent.glob("*.png")), None)
    if img_path is None:
        return []
    with open(json_path) as f:
        instances = json.load(f)
    polygons = []
    for inst in instances:
        for contour in [inst.get("contour", [])] + inst.get("extra_contours", []):
            pts = np.asarray(contour, dtype=np.float64).reshape(-1, 2)
            if len(pts) >= 3:
                polygons.append([pts])
    return [{"image": img_path, "polygons": polygons}]


def mask_from_polygons(shape, polygons):
    mask = np.zeros(shape, dtype=np.uint8)
    for parts in polygons:
        cv2.fillPoly(mask, [p.round().astype(np.int32) for p in parts], 1)
    return mask


records = []
for jpath in DATA_DIR.rglob("*.json"):
    try:
        if jpath.name == "instances.json":
            records += load_instances(jpath)
        else:
            with open(jpath) as f:
                head = json.load(f)
            if "images" in head and "annotations" in head:
                records += load_coco(jpath)
    except Exception as e:
        print(f"Could not read {jpath.name}: {e}")

# an app COCO export contains both annotations.coco.json and instances.json for the same image
samples, seen = [], set()
for rec in records:
    if rec["polygons"] and rec["image"] not in seen:
        seen.add(rec["image"])
        samples.append(rec)

assert samples, "No labelled images found. Make sure the ZIP contains app exports (COCO or Instances)."

for folder in sorted(p for p in DATA_DIR.iterdir() if p.is_dir()):
    if not any(folder in s["image"].parents for s in samples):
        print(f"Note: {folder.name} contains no image with annotations — skipped.")

n_particles = sum(len(s["polygons"]) for s in samples)
print(f"Found {len(samples)} labelled images with {n_particles} particles in total.")
if len(samples) < 20:
    print("That is quite few — expect the model to mainly memorise these exact images.")

# %% [markdown]
# # Step 2 — Look at your data
#
# A few of your images with the particle outlines drawn on top. This is what the
# model learns from, so it is worth a glance to confirm the outlines sit on the particles.

# %%
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, min(4, len(samples)), figsize=(16, 4))
for ax, sample in zip(np.atleast_1d(axes), samples[:4]):
    img = cv2.imread(str(sample["image"]), cv2.IMREAD_GRAYSCALE)
    mask = mask_from_polygons(img.shape, sample["polygons"])
    overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    overlay[mask.astype(bool)] = [255, 80, 80]
    ax.imshow(overlay)
    ax.set_title(f"{len(sample['polygons'])} particles")
    ax.axis("off")
plt.tight_layout()
plt.show()

# %% [markdown]
# # Step 3 — Training settings
#
# The defaults are a sensible starting point. You can change:
#
# - **EPOCHS** — how many times the model sees all your images. 10 is fine to start;
#   20–30 if you have 100+ images and want to squeeze out more accuracy.
# - **BATCH_SIZE** — how many images the model digests at once. Keep at 2 unless you
#   know you have GPU memory to spare.
# - **LEARNING_RATE** — how big the correction steps are. Leave at 0.00005 unless
#   training looks unstable. If the loss says `nan`, lower it and rerun from Step 4.

# %%
import random

EPOCHS = 10
BATCH_SIZE = 2
LEARNING_RATE = 0.00005
VAL_FRACTION = 0.15
COPIES_PER_IMAGE = 8  # augmented variants of each image per epoch
MIN_SCORE = 0.95  # the app only shows detections above this confidence

random.shuffle(samples)
n_val = max(1, int(len(samples) * VAL_FRACTION)) if len(samples) > 6 else 0
val_samples, train_samples = samples[:n_val], samples[n_val:]
print(f"{len(train_samples)} images for training, {len(val_samples)} for checking.")

# %% [markdown]
# # Step 4 — Train
#
# Run this cell and watch the **loss** numbers. Loss is "how wrong the model is" —
# it should go down as training progresses. A flat or slowly falling loss after a
# while means the model has learned what it can from your data.
#
# Each time an image is shown to the model it is randomly flipped and rotated, so the
# model sees far more variety than the raw image count suggests.
#
# This is the long step. The cell shows progress and will keep running for a while.
# Your weights are saved automatically after every epoch, so a disconnect at the
# very end is not a disaster.

# %%
import time

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


class ParticleDataset(Dataset):
    def __init__(self, samples, copies_per_image=1):
        self.samples = samples * copies_per_image

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = cv2.imread(str(sample["image"]), cv2.IMREAD_GRAYSCALE)
        mask = mask_from_polygons(img.shape, sample["polygons"])

        # std augmentations: flips and 90-degree rotations, same on image and mask
        k = random.randrange(4)
        img = np.rot90(img, k)
        mask = np.rot90(mask, k)
        if random.random() < 0.5:
            img, mask = np.fliplr(img), np.fliplr(mask)
        if random.random() < 0.5:
            img, mask = np.flipud(img), np.flipud(mask)
        img, mask = np.ascontiguousarray(img), np.ascontiguousarray(mask)

        # same conversion the app uses: grayscale replicated to 3 channels, scaled to 0-1
        x = torch.from_numpy(np.stack([img] * 3, axis=-1)).permute(2, 0, 1).float() / 255.0

        num, labels = cv2.connectedComponents(mask)
        masks, boxes = [], []
        for i in range(1, num):
            m = (labels == i).astype(np.uint8)
            ys, xs = np.nonzero(m)
            if len(xs) == 0:
                continue
            boxes.append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])
            masks.append(m)
        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "masks": torch.as_tensor(np.array(masks, dtype=np.uint8)).reshape(-1, *img.shape),
            "labels": torch.ones(len(masks), dtype=torch.int64),
        }
        return x, target


def collate(batch):
    return tuple(zip(*batch))


def build_model():
    model = maskrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, 2)
    return model


device = torch.device("cuda")
model = build_model().to(device)
optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LEARNING_RATE)

train_loader = DataLoader(ParticleDataset(train_samples, COPIES_PER_IMAGE), batch_size=BATCH_SIZE,
                          shuffle=True, collate_fn=collate)
val_loader = DataLoader(ParticleDataset(val_samples), batch_size=BATCH_SIZE,
                        shuffle=False, collate_fn=collate) if val_samples else None

best_val = float("inf")
for epoch in range(1, EPOCHS + 1):
    model.train()
    t0, running, n_batches = time.time(), 0.0, 0
    for images, targets in train_loader:
        images = [im.to(device) for im in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running += loss.item()
        n_batches += 1
        print(f"\r  epoch {epoch}/{EPOCHS}  batch {n_batches}  loss {loss.item():.3f}", end="")
    train_loss = running / max(n_batches, 1)

    val_loss = float("nan")
    if val_loader is not None:
        model.train()  # losses are only computed in training mode
        with torch.no_grad():
            vals = [sum(model([im.to(device) for im in images],
                              [{k: v.to(device) for k, v in t.items()} for t in targets]).values()).item()
                    for images, targets in val_loader]
        val_loss = sum(vals) / len(vals)

    score = val_loss if val_loader is not None else train_loss
    torch.save(model.state_dict(), "maskrcnn_custom.pth")
    if score < best_val:
        best_val = score
        torch.save(model.state_dict(), "maskrcnn_best.pth")
    print(f"\r  epoch {epoch}/{EPOCHS}   training loss {train_loss:.3f}   checking loss {val_loss:.3f}   ({time.time() - t0:.0f}s)")

print(f"\nDone. Best loss so far: {best_val:.3f}")

# %% [markdown]
# # Step 5 — See how it did
#
# Below, the model looks at images from your data and draws what it detects
# (red = particles it found). These are images it learned from, so this shows it
# can reproduce your labels — it will be more cautious on brand-new images.
#
# "Best confidence" is how sure the model is about its most confident detection.
# The app only accepts detections above 0.95, so a number below that means the model
# is not ready yet — train for more epochs or add more labelled images.
#
# The cell also reloads the saved weights file exactly the way the TEMseg app loads it.
# If it prints "Weights file is ready", you are good to go.

# %%
import matplotlib.pyplot as plt

device = torch.device("cuda")
model = build_model().to(device)

# reload from disk the same way the app does: torch.load then load_state_dict
model.load_state_dict(torch.load("maskrcnn_best.pth", map_location=device))
model.eval()
print("Weights file is ready for the app.")

fig, axes = plt.subplots(1, min(4, len(samples)), figsize=(16, 4))
for ax, sample in zip(np.atleast_1d(axes), samples[:4]):
    img = cv2.imread(str(sample["image"]), cv2.IMREAD_GRAYSCALE)
    x = torch.from_numpy(np.stack([img] * 3, axis=-1)).permute(2, 0, 1).float() / 255.0
    with torch.no_grad():
        pred = model([x.to(device)])[0]
    keep = pred["scores"] > MIN_SCORE
    top = float(pred["scores"].max()) if len(pred["scores"]) else 0.0
    canvas = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    for m in pred["masks"][keep].cpu().numpy():
        canvas[m[0] > 0.5] = [255, 80, 80]
    ax.imshow(canvas)
    ax.set_title(f"found {int(keep.sum())} of {len(sample['polygons'])} (best confidence {top:.2f})")
    ax.axis("off")
plt.tight_layout()
plt.show()

# %% [markdown]
# # Step 6 — Download your weights
#
# Run the cell and save the `maskrcnn_best.pth` file somewhere you can find it —
# usually your Downloads folder.

# %%
from google.colab import files

files.download("maskrcnn_best.pth")

# %% [markdown]
# # Step 7 — Use your model in the app
#
# 1. Rename the downloaded file to **`maskrcnn_best_model.pth`**.
# 2. Put it in the app's weights folder, replacing the file that is already there:
#    - **Mac:** `~/Library/Application Support/TEMseg/weights/`
#    - **Windows:** `%LOCALAPPDATA%\TEMseg\weights\`
#    - **Linux:** `~/.local/share/TEMseg/weights/`
#
#    If you would rather keep the built-in model, put your file anywhere and start
#    the app with the environment variable `TEMSEG_WEIGHTS_DIR` pointing at that folder.
# 3. Restart the app and run the Mask R-CNN model as usual.
#
# **Good to know**
#
# - The app only shows detections with confidence above **0.95** (a very high bar on
#   purpose). If your model finds too little, your particles look too different from
#   the training data — add a few more labelled images and retrain.
# - The numbers in `stats.csv` and the app are computed from the final masks, so they
#   stay exactly as accurate with a custom model.
# - Training images should look like the images you will segment in the app (similar
#   magnification and contrast). A model trained on one kind of image does not
#   magically work on a very different kind.
