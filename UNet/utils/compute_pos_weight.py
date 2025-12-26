import argparse
from pathlib import Path

from PIL import Image
import numpy as np
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


def find_mask(mask_dir: Path, img_name: str) -> Path:
    stem = Path(img_name).stem
    for s in (stem, f"{stem}_mask"):
        for ext in (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff"):
            cand = mask_dir / f"{s}{ext}"
            if cand.exists():
                return cand
    found = list(mask_dir.glob(f"{stem}*.*"))
    if found:
        return found[0]
    raise FileNotFoundError(f"mask not found for {img_name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="UNet/data")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--max", type=int, default=0, help="max samples to scan; 0 = all")
    args = parser.parse_args()

    root = Path(args.root)
    img_dir = root / args.split / "images"
    mask_dir = root / args.split / "masks"

    img_files = sorted([p for p in img_dir.iterdir() if p.is_file()])
    if not img_files:
        raise RuntimeError("no images found")
    if args.max > 0:
        img_files = img_files[: args.max]

    fg_sum = 0.0
    total = 0.0
    it = img_files
    if tqdm is not None:
        it = tqdm(img_files, desc="Scanning masks", unit="img")
    for img_path in it:
        mask_path = find_mask(mask_dir, img_path.name)
        mask = Image.open(mask_path).convert("L")
        arr = np.array(mask, dtype=np.uint8)
        fg = (arr > 127).sum()
        fg_sum += fg
        total += arr.size

    fg_ratio = fg_sum / total if total > 0 else 0.0
    if fg_ratio <= 0:
        print("foreground ratio: 0.0 (check masks)")
        return
    pos_weight = (1.0 - fg_ratio) / fg_ratio
    print(f"foreground ratio: {fg_ratio:.6f}")
    print(f"suggested pos_weight: {pos_weight:.2f}")


if __name__ == "__main__":
    main()
