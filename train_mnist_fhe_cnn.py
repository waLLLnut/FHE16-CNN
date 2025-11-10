import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# ────────────────────────────────────────────────
# Model: Conv(1→3, 3x3, stride=3, bias=True)
#         → ReLU → SumPool(3x3, stride=3) → Flatten(27) → FC(27→10, bias=True)
# ────────────────────────────────────────────────
class CNN_3x3_stride3(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 3, kernel_size=3, stride=3, padding=0, bias=True)
        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=3)  # SumPool = AvgPool * 9
        self.fc = nn.Linear(3 * 3 * 3, 10, bias=True)         # 27 → 10

    def forward(self, x):
        x = self.conv(x)                # [B,3,9,9]
        x = F.relu(x, inplace=True)
        x = self.avgpool(x) * 9.0       # SumPool(3x3)
        x = torch.flatten(x, 1)         # [B,27]
        x = self.fc(x)                  # [B,10]
        return x


# ────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / total


@torch.no_grad()
def to_int_scaled(t: torch.Tensor, scale: int) -> torch.Tensor:
    return torch.round(t * scale).to(torch.int32)


@torch.no_grad()
def export_conv_csv(path_w: Path, path_b: Path, conv: nn.Conv2d, scale: int):
    W = to_int_scaled(conv.weight.detach().cpu(), scale)  # [3,1,3,3]
    B = to_int_scaled(conv.bias.detach().cpu(),   scale)  # [3]
    vals_w = []
    for k in range(W.shape[0]):
        for dy in range(W.shape[2]):
            for dx in range(W.shape[3]):
                vals_w.append(str(int(W[k, 0, dy, dx])))
    path_w.write_text(",".join(vals_w) + "\n")
    path_b.write_text(",".join([str(int(b)) for b in B.tolist()]) + "\n")


@torch.no_grad()
def export_fc_csv(path_w: Path, path_b: Path, fc: nn.Linear, scale: int):
    W = to_int_scaled(fc.weight.detach().cpu(), scale)  # [10,27]
    B = to_int_scaled(fc.bias.detach().cpu(),   scale)  # [10]
    vals_w = []
    for c in range(W.shape[0]):
        for i in range(W.shape[1]):
            vals_w.append(str(int(W[c, i])))
    path_w.write_text(",".join(vals_w) + "\n")
    path_b.write_text(",".join([str(int(b)) for b in B.tolist()]) + "\n")


# ────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out-dir", default="./export_csv")
    ap.add_argument("--scale", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=2)
    args = ap.parse_args()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda t: t * 255.0),
    ])
    train_ds = datasets.MNIST(args.data_root, train=True,  download=True, transform=transform)
    test_ds  = datasets.MNIST(args.data_root, train=False, download=True, transform=transform)
    train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=True)
    test_ld  = DataLoader(test_ds, batch_size=512, shuffle=False,
                          num_workers=args.num_workers, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = CNN_3x3_stride3().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss()

    # ── Train
    for ep in range(1, args.epochs + 1):
        net.train()
        for x, y in train_ld:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = crit(net(x), y)
            loss.backward()
            opt.step()
        acc = evaluate(net, test_ld, device)
        print(f"[Epoch {ep}] test acc (float) = {acc*100:.2f}%")

    # ── Quantization check
    with torch.no_grad():
        qnet = CNN_3x3_stride3().to(device)
        qnet.load_state_dict(net.state_dict())
        for m in qnet.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                m.weight.copy_(torch.round(m.weight * args.scale) / args.scale)
                if m.bias is not None:
                    m.bias.copy_(torch.round(m.bias * args.scale) / args.scale)
        qacc = evaluate(qnet, test_ld, device)
        print(f"[Eval] quantized acc (scale={args.scale}) = {qacc*100:.2f}%")

    # ── Export weights
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    export_conv_csv(out/"conv_w.csv", out/"conv_b.csv", net.conv, args.scale)
    export_fc_csv(out/"fc_w.csv", out/"fc_b.csv", net.fc, args.scale)
    print("Saved:", out/"conv_w.csv", out/"conv_b.csv", out/"fc_w.csv", out/"fc_b.csv")

    # ── Sanity test
    x, y = next(iter(test_ld))
    with torch.no_grad():
        pred = net(x.to(device)).argmax(1).cpu()
    print(f"[Sanity] batch acc = {(pred == y).float().mean().item()*100:.2f}%")


if __name__ == "__main__":
    main()

