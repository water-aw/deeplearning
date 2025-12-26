import argparse
import torch
from PIL import Image
from torchvision import transforms
from unet import UNet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="best.pt")
    parser.add_argument("--image", type=str, default="your_image.jpg")
    parser.add_argument("--size", type=int, nargs=2, default=[384, 640], metavar=("H", "W"))
    parser.add_argument("--thresh", type=float, default=0.05, help="mask threshold")
    parser.add_argument("--out_mask", type=str, default="pred_mask.png")
    parser.add_argument("--out_prob", type=str, default="pred_prob.png")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(args.ckpt, map_location=device)
    cfg = ckpt["cfg"]
    model = UNet(n_channels=cfg["in_channels"], n_classes=cfg["num_classes"], base_c=cfg["base_channels"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    img = Image.open(args.image).convert("RGB")
    tfm = transforms.Compose([transforms.Resize(tuple(args.size)), transforms.ToTensor()])
    x = tfm(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        pred = torch.sigmoid(logits)
        prob = pred[0, 0].cpu().numpy()
        mask = (pred > args.thresh).float()
        mask = mask[0, 0].cpu().numpy()

    Image.fromarray((prob * 255).astype("uint8")).save(args.out_prob)
    Image.fromarray((mask * 255).astype("uint8")).save(args.out_mask)
    print(f"saved {args.out_prob}")
    print(f"saved {args.out_mask}")
    print(pred.min().item(), pred.max().item(), pred.mean().item())


if __name__ == "__main__":
    main()
