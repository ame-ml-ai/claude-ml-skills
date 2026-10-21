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
shift_norm = 3.0
seeds = (0, 1, 2)
lr_grid = {"full": (1e-3, 3e-3, 1e-2), "lora": (3e-3, 1e-2, 3e-2)}
ranks = (1, 4, 8)
steps = 600


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        d, k = base.out_features, base.in_features
        self.r = r
        self.scaling = alpha / r
        # gaussian A, zero B, so delta_w = BA = 0 at start
        self.A = nn.Parameter(torch.randn(r, k, device=base.weight.device) / math.sqrt(k))
        self.B = nn.Parameter(torch.zeros(d, r, device=base.weight.device))

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


def add_lora(model, r, alpha=8):
    model = copy.deepcopy(model)
    for p in model.parameters():
        p.requires_grad_(False)
    model.fc1 = LoRALinear(model.fc1, r, alpha)
    model.fc2 = LoRALinear(model.fc2, r, alpha)
    return model


def planted_shift(layer, rank, norm):
    d, k = layer.weight.shape
    u, v = torch.randn(d, rank, device=device), torch.randn(rank, k, device=device)
    dw = u @ v
    dw *= norm / dw.norm()
    layer.weight.data += dw
    return dw


def make_tasks(shift_rank):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        teacher_a = MLP().to(device)
        with torch.no_grad():
            # rescale so task a targets have unit variance
            scale = teacher_a(torch.randn(4096, d_in, device=device)).std()
            teacher_a.fc3.weight /= scale
            teacher_a.fc3.bias /= scale
        teacher_b = copy.deepcopy(teacher_a)
        with torch.no_grad():
            dw1 = planted_shift(teacher_b.fc1, shift_rank, shift_norm)
            planted_shift(teacher_b.fc2, shift_rank, shift_norm)
    return teacher_a, teacher_b, dw1


def make_data(teacher, n):
    x = torch.randn(n, d_in, device=device)
    with torch.no_grad():
        y = teacher(x)
    return x, y


def n_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def mse(model, x, y):
    with torch.no_grad():
        return F.mse_loss(model(x), y).item()


def train(model, x, y, steps, lr, batch=128):
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    for _ in range(steps):
        idx = torch.randint(0, len(x), (batch,), device=device)
        loss = F.mse_loss(model(x[idx]), y[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


def build(method, pretrained):
    return copy.deepcopy(pretrained) if method == "full" else add_lora(pretrained, method)


def fit(method, pretrained, data, lr, seed):
    torch.manual_seed(seed)
    model = build(method, pretrained)
    return train(model, data["x_train"], data["y_train"], steps, lr)


def fc1_delta(model, pretrained):
    if isinstance(model.fc1, LoRALinear):
        return model.fc1.delta_w().detach()
    return (model.fc1.weight - pretrained.fc1.weight).detach()


def subspace_similarity(m1, m2, i, j):
    # eq. 4 of section 7.2: ||U1_i^T U2_j||_F^2 / min(i, j), with right singular vectors
    u1 = torch.linalg.svd(m1, full_matrices=False).Vh[:i].T
    u2 = torch.linalg.svd(m2, full_matrices=False).Vh[:j].T
    return (u1.T @ u2).pow(2).sum().item() / min(i, j)


def test():
    torch.manual_seed(1)
    base = MLP().to(device)
    x = torch.randn(16, d_in, device=device)

    lora = add_lora(base, r=4)
    assert lora.fc1.A.shape == (4, d_in) and lora.fc1.B.shape == (d_hidden, 4)
    assert lora.fc1.delta_w().shape == base.fc1.weight.shape
    assert lora(x).shape == (16, d_out)
    assert n_trainable(lora) == 4 * (d_in + d_hidden) + 4 * (d_hidden + d_hidden)

    # zero B, so the adapted model starts exactly at the pretrained one
    assert torch.equal(lora(x), base(x))

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
    assert torch.linalg.matrix_rank(lora.fc1.delta_w()) == 4

    # merged W = W0 + (alpha/r) BA gives the same output with plain linear layers
    merged = copy.deepcopy(lora)
    merged.fc1, merged.fc2 = lora.fc1.merge(), lora.fc2.merge()
    assert torch.allclose(merged(x), lora(x), atol=1e-5)

    a = torch.randn(4, 32)
    assert math.isclose(subspace_similarity(a, 3 * a, 4, 4), 1.0, rel_tol=1e-5)
    q = torch.linalg.qr(torch.randn(32, 8)).Q
    assert subspace_similarity(q[:, :4].T, q[:, 4:].T, 4, 4) < 1e-10
    print(f"tests passed (overfit loss {loss.item():.1e})")


def run_task(shift_rank, lrs=None):
    teacher_a, teacher_b, dw1_true = make_tasks(shift_rank)
    torch.manual_seed(100 + shift_rank)
    data = {}
    for split, n in (("train", 1000), ("val", 1000), ("test", 2000)):
        data[f"x_{split}"], data[f"y_{split}"] = make_data(teacher_b, n)
    pretrained = teacher_a
    methods = ["full"] + list(ranks)
    if lrs is None:
        lrs = {}
        for method in methods:
            grid = lr_grid["full" if method == "full" else "lora"]
            val = {lr: mse(fit(method, pretrained, data, lr, 0), data["x_val"], data["y_val"]) for lr in grid}
            lrs[method] = min(val, key=val.get)
    results, models = {}, {}
    for method in methods:
        runs = [fit(method, pretrained, data, lrs[method], s) for s in seeds]
        losses = torch.tensor([mse(m, data["x_test"], data["y_test"]) for m in runs])
        results[method] = (losses.mean().item(), losses.std().item(), n_trainable(runs[0]))
        models[method] = runs[0]
    results["none"] = (mse(pretrained, data["x_test"], data["y_test"]), 0.0, 0)
    return results, models, pretrained, dw1_true, lrs


def name(method):
    return {"full": "full fine-tuning", "none": "no adaptation"}.get(method, f"LoRA r={method}")


def plot(low, high, spectra, path):
    try:
        import matplotlib
    except ImportError:
        print("matplotlib not installed, skipping the plot")
        return
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink = "#8a8a86"
    blue, orange, green = "#2a78d6", "#eb6834", "#1baf7a"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8), dpi=150)

    methods = ["none", "full"] + list(ranks)
    xs = range(len(methods))
    for offset, (res, color, label) in enumerate(((low, blue, "rank-4 shift"), (high, orange, "full-rank shift"))):
        means = [res[m][0] for m in methods]
        stds = [res[m][1] for m in methods]
        ax1.bar([x + (offset - 0.5) * 0.38 for x in xs], means, 0.38, yerr=stds, color=color,
                label=label, error_kw={"ecolor": ink, "capsize": 2, "lw": 1})
    ax1.set_xticks(list(xs), ["none", "full FT"] + [f"r={r}" for r in ranks])
    ax1.set_yscale("log")
    ax1.set_ylabel("test MSE on task B (log)", color=ink)
    ax1.set_title("same shift norm, different shift rank", color=ink, fontsize=10)
    ax1.legend(frameon=False, labelcolor=ink, fontsize=8)

    k = 10
    idx = range(1, k + 1)
    for key, color, marker, label in (("true", ink, "o", "planted ΔW*"), ("full", blue, "s", "full fine-tuning"),
                                      (8, green, "^", "LoRA r=8")):
        sv = spectra[key][:k].cpu()
        ax2.plot(idx, sv, color=color, marker=marker, ms=4, lw=1.5, label=label)
    ax2.set_xlabel("singular value index", color=ink)
    ax2.set_ylabel("singular value of ΔW (fc1)", color=ink)
    ax2.set_title("learned update vs planted rank-4 update", color=ink, fontsize=10)
    ax2.legend(frameon=False, labelcolor=ink, fontsize=8)

    for ax in (ax1, ax2):
        ax.tick_params(which="both", colors=ink)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(ink)
        ax.grid(axis="y", color=ink, alpha=0.2)
    fig.tight_layout()
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path, transparent=True)
    print(f"saved {path.name}")


if __name__ == "__main__":
    test()
    if "--test" in sys.argv:
        sys.exit()
    start = time.time()

    low, models, pretrained, dw1_true, lrs = run_task(true_rank)
    high, *_ = run_task(min(d_in, d_hidden), lrs)

    print(f"\ntest MSE, mean ± std over {len(seeds)} seeds, lr picked on a validation set")
    print(f"{'method':<18}{'params':>8}{'lr':>8}{'rank-4 shift':>18}{'full-rank shift':>18}")
    for method in ["none", "full"] + list(ranks):
        lr = f"{lrs[method]:.0e}" if method in lrs else "-"
        cells = "".join(f"{r[method][0]:>11.4f} ± {r[method][1]:.4f}" for r in (low, high))
        print(f"{name(method):<18}{low[method][2]:>8}{lr:>8}{cells}")

    print("\nrank-4 shift, fc1: does the learned ΔW find the planted ΔW*?")
    print(f"{'method':<18}{'rel. error':>11}{'φ(ΔW, ΔW*)':>13}   top singular values")
    spectra = {"true": torch.linalg.svdvals(dw1_true)}
    for method in ["full"] + list(ranks):
        dw = fc1_delta(models[method], pretrained)
        spectra[method] = torch.linalg.svdvals(dw)
        err = ((dw - dw1_true).norm() / dw1_true.norm()).item()
        i = true_rank if method == "full" else min(true_rank, method)
        phi = subspace_similarity(dw, dw1_true, i, true_rank)
        top = " ".join(f"{s:.2f}" for s in spectra[method][:6].tolist())
        print(f"{name(method):<18}{err:>11.3f}{phi:>13.3f}   {top}")
    print(f"{'planted ΔW*':<18}{'':>11}{'':>13}   "
          + " ".join(f"{s:.2f}" for s in spectra["true"][:6].tolist()))

    print(f"\ntotal time {time.time() - start:.1f}s")
    plot(low, high, spectra, Path(__file__).parent / "assets" / "results.png")
