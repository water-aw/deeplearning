import argparse
import random
from pathlib import Path

from PIL import Image, ImageColor
from torchvision.transforms import functional as F
from torchvision.transforms import InterpolationMode


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
    parser.add_argument("--name", type=str, default="", help="image filename; empty -> random")
    parser.add_argument("--size", type=int, nargs=2, default=[384, 640], metavar=("H", "W"))
    parser.add_argument("--alpha", type=float, default=0.5, help="overlay opacity")
    parser.add_argument("--color", type=str, default="#ff0000", help="mask color")
    parser.add_argument("--out", type=str, default="overlay.png")
    args = parser.parse_args()

    root = Path(args.root)
    img_dir = root / args.split / "images"
    mask_dir = root / args.split / "masks"

    img_files = sorted([p for p in img_dir.iterdir() if p.is_file()])
    if not img_files:
        raise RuntimeError("no images found")

    if args.name:
        img_path = img_dir / args.name
    else:
        img_path = random.choice(img_files)

    mask_path = find_mask(mask_dir, img_path.name)

    img = Image.open(img_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")
    img = F.resize(img, args.size)
    mask = F.resize(mask, args.size, interpolation=InterpolationMode.NEAREST)

    # binarize mask for display
    mask_bin = mask.point(lambda p: 255 if p > 127 else 0)
    color = Image.new("RGB", img.size, ImageColor.getrgb(args.color))
    overlay = Image.composite(color, img, mask_bin)
    blended = Image.blend(img, overlay, alpha=args.alpha)

    blended.save(args.out)
    print(f"saved {args.out} from {img_path.name} and {mask_path.name}")


if __name__ == "__main__":
    main()
