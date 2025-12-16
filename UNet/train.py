import torch, argparse
from torch.utils.data import DataLoader
from unet import UNet
from loss import build_loss
from metrics import dice_coef, iou
from data import build_datasets
from config import cfg

def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total = 0.0
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = loss_fn(logits, masks)
        loss.backward()
        optimizer.step()
        total += loss.item() * imgs.size(0)
    return total / len(loader.dataset)

@torch.no_grad()
def eval_one_epoch(model, loader, loss_fn, device):
    model.eval()
    total, dices, ious = 0.0, [], []
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        logits = model(imgs)
        loss = loss_fn(logits, masks)
        total += loss.item() * imgs.size(0)
        dices.append(dice_coef(logits, masks).item())
        ious.append(iou(logits, masks).item())
    return total / len(loader.dataset), sum(dices)/len(dices), sum(ious)/len(ious)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=cfg.epochs)
    parser.add_argument("--lr", type=float, default=cfg.lr)
    parser.add_argument("--batch", type=int, default=cfg.batch_size)
    parser.add_argument("--loss", type=str, default=cfg.loss_name)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_ds, val_ds = build_datasets(cfg)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=cfg.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=cfg.num_workers)

    model = UNet(n_channels=cfg.in_channels, n_classes=cfg.num_classes, base_c=cfg.base_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.lr_step, gamma=cfg.lr_gamma)
    loss_fn = build_loss(args.loss)

    best_dice = 0.0
    for epoch in range(args.epochs):
        tr_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        va_loss, va_dice, va_iou = eval_one_epoch(model, val_loader, loss_fn, device)
        scheduler.step()
        print(f"Epoch {epoch+1}: train {tr_loss:.4f} | val {va_loss:.4f} | dice {va_dice:.4f} | iou {va_iou:.4f}")
        if va_dice > best_dice:
            best_dice = va_dice
            torch.save({"model": model.state_dict(), "cfg": cfg.__dict__}, "best.pt")

if __name__ == "__main__":
    main()


# python train.py --loss bce+dice --epochs 50 --batch 4