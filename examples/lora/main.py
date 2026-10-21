import copy
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"

d_in, d_hidden, d_out = 32, 128, 8
true_rank = 4


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        d, k = base.out_features, base.in_features
        self.r = r
        self.scaling = alpha / r
        #gaussian A, zero B, so delta_w = BA = 0 at start
        self.A = nn.Parameter(torch.randn(r, k) / math.sqrt(k))
        self.B = nn.Parameter(torch.zeros(d, r))

    def delta_w(self):
        return self.scaling * self.B @ self.A

    def forward(self, x):
        return self.base(x) + self.scaling * (x @ self.A.T) @ self.B.T

    def merge(self):
        merged = nn.Linear(self.base.in_features, self.base.out_features, device=self.A.device)
        merged.weight.data = self.base.weight + self.delta_w()
        merged.bias.data = self.base.bias.clone()
        return merged


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_hidden)
        self.fc3 = nn.Linear(d_hidden, d_out)

    def forward(self, x):
        return self.fc3(F.gelu(self.fc2(F.gelu(self.fc1(x)))))


def add_lora(model, r, alpha):
    model = copy.deepcopy(model)
    for p in model.parameters():
        p.requires_grad_(False)
    model.fc1 = LoRALinear(model.fc1, r, alpha)
    model.fc2 = LoRALinear(model.fc2, r, alpha)
    return model


def low_rank_shift(layer, rank, size):
    d, k = layer.weight.shape
    u = torch.randn(d, rank) / math.sqrt(d)
    v = torch.randn(rank, k) / math.sqrt(k)
    layer.weight.data += size * u @ v


def make_tasks():
    teacher_a = MLP().to(device)
    with torch.no_grad():
        #rescale so task a targets have unit variance
        scale = teacher_a(torch.randn(4096, d_in, device=device)).std()
        teacher_a.fc3.weight /= scale
        teacher_a.fc3.bias /= scale
    #task b = task a with a rank-4 change in the first two layers
    teacher_b = copy.deepcopy(teacher_a)
    with torch.no_grad():
        low_rank_shift(teacher_b.fc1, true_rank, 1.5)
        low_rank_shift(teacher_b.fc2, true_rank, 1.5)
    return teacher_a, teacher_b


def make_data(teacher, n):
    x = torch.randn(n, d_in, device=device)
    with torch.no_grad():
        y = teacher(x)
    return x, y


def n_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train(model, x, y, x_test, y_test, steps, lr, batch=128):
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    curve = []
    for step in range(steps):
        idx = torch.randint(0, len(x), (batch,), device=device)
        loss = F.mse_loss(model(x[idx]), y[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 20 == 0 or step == steps - 1:
            with torch.no_grad():
                curve.append((step, F.mse_loss(model(x_test), y_test).item()))
    return curve


def test():
    torch.manual_seed(1)
    base = MLP().to(device)
    x = torch.randn(16, d_in, device=device)

    lora = add_lora(base, r=4, alpha=8)
    assert lora.fc1.A.shape == (4, d_in) and lora.fc1.B.shape == (d_hidden, 4)
    assert lora.fc1.delta_w().shape == base.fc1.weight.shape
    assert lora(x).shape == (16, d_out)

    #zero B, so the adapted model starts exactly at the pretrained one
    assert torch.allclose(lora(x), base(x))

    trainable = {n for n, p in lora.named_parameters() if p.requires_grad}
    assert trainable == {"fc1.A", "fc1.B", "fc2.A", "fc2.B"}, trainable

    y = torch.randn(16, d_out, device=device)
    opt = torch.optim.Adam([p for p in lora.parameters() if p.requires_grad], lr=1e-2)
    w0 = base.fc1.weight.clone()
    for _ in range(300):
        loss = F.mse_loss(lora(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 1e-2, loss.item()
    assert torch.equal(lora.fc1.base.weight, w0)

    assert torch.linalg.matrix_rank(lora.fc1.delta_w()) <= 4

    #merged W = W0 + (alpha/r) BA gives the same output with plain linear layers
    merged = copy.deepcopy(lora)
    merged.fc1, merged.fc2 = lora.fc1.merge(), lora.fc2.merge()
    assert torch.allclose(merged(x), lora(x), atol=1e-5)
    print(f"tests passed (overfit loss {loss.item():.1e})")


def plot(curves, path):
    try:
        import matplotlib
    except ImportError:
        print("matplotlib not installed, skipping the plot")
        return
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink = "#8a8a86"
    colors = {"full fine-tuning": "#2a78d6", "LoRA r=1": "#eb6834", "LoRA r=4": "#1baf7a"}
    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    for name, color in colors.items():
        steps, losses = zip(*curves[name])
        ax.plot(steps, losses, color=color, lw=2, label=name)
        ax.annotate(name, (steps[-1], losses[-1]), xytext=(6, 0), textcoords="offset points",
                    color=ink, va="center", fontsize=9)
    ax.axhline(curves["frozen"], color=ink, lw=1.5, ls="--", label="no adaptation")
    ax.set_yscale("log")
    ax.set_xlabel("training step on task B", color=ink)
    ax.set_ylabel("test MSE on task B (log)", color=ink)
    ax.set_title("LoRA vs full fine-tuning on a rank-4 task shift", color=ink)
    ax.tick_params(which="both", colors=ink)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(ink)
    ax.grid(axis="y", color=ink, alpha=0.2)
    ax.set_ylim(top=1.2)
    ax.legend(frameon=False, labelcolor=ink, loc="upper right", ncol=2)
    fig.tight_layout()
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path, transparent=True)
    print(f"saved {path.name}")


if __name__ == "__main__":
    test()
    if "--test" in sys.argv:
        sys.exit()
    start = time.time()

    teacher_a, teacher_b = make_tasks()
    x_a, y_a = make_data(teacher_a, 4000)
    x_b, y_b = make_data(teacher_b, 1000)
    x_test, y_test = make_data(teacher_b, 2000)

    #pretraining on task a
    pretrained = MLP().to(device)
    train(pretrained, x_a, y_a, *make_data(teacher_a, 500), steps=2000, lr=3e-3)

    with torch.no_grad():
        frozen = F.mse_loss(pretrained(x_test), y_test).item()
    curves = {"frozen": frozen}
    print(f"{'method':<18}{'trainable params':>18}{'test MSE':>12}")
    print(f"{'no adaptation':<18}{0:>18}{frozen:>12.4f}")

    #adaptation to task b
    runs = [("full fine-tuning", copy.deepcopy(pretrained), 1e-3)]
    runs += [(f"LoRA r={r}", add_lora(pretrained, r, alpha=8), 1e-2) for r in (1, 2, 4, 8)]
    for name, model, lr in runs:
        torch.manual_seed(0)
        curve = train(model, x_b, y_b, x_test, y_test, steps=600, lr=lr)
        curves[name] = curve
        print(f"{name:<18}{n_trainable(model):>18}{curve[-1][1]:>12.4f}")

    print(f"total time {time.time() - start:.1f}s")
    plot(curves, Path(__file__).parent / "assets" / "results.png")
