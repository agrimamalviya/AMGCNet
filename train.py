import argparse, os, random
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from amgcnet.models.amgcnet import AMGCNet
import yaml

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def get_loaders(cfg):
    tfm = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    train_ds = datasets.ImageFolder(cfg["data"]["train_dir"], transform=tfm)
    val_ds   = datasets.ImageFolder(cfg["data"]["val_dir"],   transform=tfm)
    train_dl = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True,
                          num_workers=cfg["data"]["num_workers"], pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=cfg["train"]["batch_size"], shuffle=False,
                          num_workers=cfg["data"]["num_workers"], pin_memory=True)
    return train_dl, val_dl

def train_one_epoch(model, dl, opt, scaler, device, loss_fn):
    model.train()
    total = 0.0
    for x, y in dl:
        x, y = x.to(device), y.to(device)
        opt.zero_grad(set_to_none=True)
        if scaler:
            with torch.cuda.amp.autocast():
                logits = model(x)
                loss = loss_fn(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
        else:
            loss = loss_fn(model(x), y)
            loss.backward(); opt.step()
        total += loss.item()
    return total / max(1, len(dl))

@torch.no_grad()
def evaluate(model, dl, device, loss_fn):
    model.eval()
    total, correct, n = 0.0, 0, 0
    for x, y in dl:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = loss_fn(logits, y)
        total += loss.item()
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        n += y.numel()
    return total / max(1, len(dl)), correct / max(1, n)

def main(args):
    cfg = yaml.safe_load(open(args.config))
    set_seed(cfg["seed"])
    device = torch.device(cfg["train"]["device"] if torch.cuda.is_available() else "cpu")

    model = AMGCNet(
        in_channels=cfg["model"]["in_channels"],
        num_classes=cfg["model"]["num_classes"],
        width=cfg["model"]["width"],
    ).to(device)

    train_dl, val_dl = get_loaders(cfg)
    opt = optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    loss_fn = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=cfg["train"]["amp"] and device.type == "cuda")

    os.makedirs(cfg["train"]["ckpt_dir"], exist_ok=True)
    best_acc, best_path = 0.0, None

    for epoch in range(cfg["train"]["epochs"]):
        tr_loss = train_one_epoch(model, train_dl, opt, scaler, device, loss_fn)
        val_loss, val_acc = evaluate(model, val_dl, device, loss_fn)

        print(f"Epoch {epoch+1}/{cfg['train']['epochs']} | train_loss={tr_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        # save best
        if val_acc > best_acc:
            best_acc = val_acc
            best_path = os.path.join(cfg["train"]["ckpt_dir"], "amgcnet_best.pt")
            torch.save({"state_dict": model.state_dict(), "acc": best_acc}, best_path)

    print(f"Best val acc: {best_acc:.4f} | ckpt: {best_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    main(parser.parse_args())

