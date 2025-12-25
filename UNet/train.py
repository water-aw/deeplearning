import torch, argparse
from datetime import datetime
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None
from torch.utils.data import DataLoader
from unet import UNet
from loss import build_loss
from metrics import dice_coef, iou
from data import build_datasets
from config import cfg
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

def train_one_epoch(model, loader, optimizer, loss_fn, device, epoch=0):
    model.train()
    total = 0.0
    it = loader
    if tqdm is not None:
        it = tqdm(loader, desc=f"Train {epoch+1}", leave=False)
    for imgs, masks in it:
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = loss_fn(logits, masks)
        loss.backward()
        optimizer.step()
        total += loss.item() * imgs.size(0)
        if tqdm is not None:
            it.set_postfix(loss=f"{loss.item():.4f}")
    return total / len(loader.dataset)

@torch.no_grad()
def eval_one_epoch(model, loader, loss_fn, device, epoch=0):
    model.eval()
    total, dices, ious = 0.0, [], []
    it = loader
    if tqdm is not None:
        it = tqdm(loader, desc=f"Val {epoch+1}", leave=False)
    for imgs, masks in it:
        imgs, masks = imgs.to(device), masks.to(device)
        logits = model(imgs)
        loss = loss_fn(logits, masks)
        total += loss.item() * imgs.size(0)
        dices.append(dice_coef(logits, masks).item())
        ious.append(iou(logits, masks).item())
        if tqdm is not None:
            it.set_postfix(loss=f"{loss.item():.4f}")
    return total / len(loader.dataset), sum(dices)/len(dices), sum(ious)/len(ious)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=cfg.epochs)
    parser.add_argument("--lr", type=float, default=cfg.lr)
    parser.add_argument("--batch", type=int, default=cfg.batch_size)
    parser.add_argument("--loss", type=str, default=cfg.loss_name)
    parser.add_argument("--logdir", type=str, default="runs", help="TensorBoard log directory")
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
    writer = None
    if SummaryWriter is not None:
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        writer = SummaryWriter(log_dir=f"{args.logdir}/{run_name}")
    for epoch in range(args.epochs):
        tr_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device, epoch)
        va_loss, va_dice, va_iou = eval_one_epoch(model, val_loader, loss_fn, device, epoch)
        scheduler.step()
        if writer is not None:
            writer.add_scalar("loss/train", tr_loss, epoch + 1)
            writer.add_scalar("loss/val", va_loss, epoch + 1)
            writer.add_scalar("metrics/dice", va_dice, epoch + 1)
            writer.add_scalar("metrics/iou", va_iou, epoch + 1)
            writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch + 1)
        print(f"Epoch {epoch+1}: train {tr_loss:.4f} | val {va_loss:.4f} | dice {va_dice:.4f} | iou {va_iou:.4f}")
        if va_dice > best_dice:
            best_dice = va_dice
            torch.save({"model": model.state_dict(), "cfg": dict(cfg.__dict__)}, "best.pt")
    if writer is not None:
        writer.close()

if __name__ == "__main__":
    main()


# python deeplearning\UNet\train.py --loss bce+dice --epochs 50 --batch 4
