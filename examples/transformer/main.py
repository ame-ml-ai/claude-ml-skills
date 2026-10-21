import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"

n_symbols, seq_len = 10, 8
bos = n_symbols
vocab = n_symbols + 1


def attention(q, k, v, mask=None, scale=True):
    d_k = q.size(-1)
    scores = q @ k.transpose(-2, -1)
    if scale:
        scores = scores / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    weights = scores.softmax(dim=-1)
    return weights @ v, weights


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, h):
        super().__init__()
        self.h, self.d_k = h, d_model // h
        self.scale = True
        # all h heads in one matrix: columns i*d_k:(i+1)*d_k are W_i^Q
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

    def split(self, x):
        b, t, _ = x.shape
        return x.view(b, t, self.h, self.d_k).transpose(1, 2)

    def forward(self, q, k, v, mask=None):
        q, k, v = self.split(self.w_q(q)), self.split(self.w_k(k)), self.split(self.w_v(v))
        # (B, h, T_q, d_k)
        heads, self.weights = attention(q, k, v, mask, self.scale)
        b, _, t, _ = heads.shape
        return self.w_o(heads.transpose(1, 2).reshape(b, t, self.h * self.d_k))


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.w_2(F.relu(self.w_1(x)))


class EncoderLayer(nn.Module):
    def __init__(self, d_model, h, d_ff, p_drop):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, h)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1, self.norm2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.drop = nn.Dropout(p_drop)

    def forward(self, x, src_mask):
        # post-ln: LayerNorm(x + Sublayer(x))
        x = self.norm1(x + self.drop(self.self_attn(x, x, x, src_mask)))
        return self.norm2(x + self.drop(self.ffn(x)))


class DecoderLayer(nn.Module):
    def __init__(self, d_model, h, d_ff, p_drop):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, h)
        self.cross_attn = MultiHeadAttention(d_model, h)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1, self.norm2, self.norm3 = (nn.LayerNorm(d_model) for _ in range(3))
        self.drop = nn.Dropout(p_drop)

    def forward(self, y, memory, tgt_mask):
        y = self.norm1(y + self.drop(self.self_attn(y, y, y, tgt_mask)))
        # queries from the decoder, keys and values from the encoder
        y = self.norm2(y + self.drop(self.cross_attn(y, memory, memory)))
        return self.norm3(y + self.drop(self.ffn(y)))


def positional_encoding(max_len, d_model):
    pos = torch.arange(max_len).unsqueeze(1)
    two_i = torch.arange(0, d_model, 2)
    angle = pos / 10000 ** (two_i / d_model)
    pe = torch.zeros(max_len, d_model)
    pe[:, 0::2] = torch.sin(angle)
    pe[:, 1::2] = torch.cos(angle)
    return pe


def causal_mask(t):
    return torch.tril(torch.ones(t, t, dtype=torch.bool, device=device))


class Transformer(nn.Module):
    def __init__(self, vocab, d_model=64, h=4, d_ff=128, n_layers=2, p_drop=0.1, max_len=64):
        super().__init__()
        self.d_model = d_model
        self.use_pe = True
        self.embed = nn.Embedding(vocab, d_model)
        # std d_model^-0.5: scaled inputs are unit size, tied logits start small
        nn.init.normal_(self.embed.weight, std=d_model ** -0.5)
        self.register_buffer("pe", positional_encoding(max_len, d_model))
        self.drop = nn.Dropout(p_drop)
        self.encoder = nn.ModuleList(EncoderLayer(d_model, h, d_ff, p_drop) for _ in range(n_layers))
        self.decoder = nn.ModuleList(DecoderLayer(d_model, h, d_ff, p_drop) for _ in range(n_layers))

    def embed_tokens(self, tokens):
        x = self.embed(tokens) * math.sqrt(self.d_model)
        if self.use_pe:
            x = x + self.pe[: tokens.size(1)]
        return self.drop(x)

    def encode(self, src):
        x = self.embed_tokens(src)
        for layer in self.encoder:
            x = layer(x, None)
        return x

    def decode(self, tgt_in, memory):
        y = self.embed_tokens(tgt_in)
        mask = causal_mask(tgt_in.size(1))
        for layer in self.decoder:
            y = layer(y, memory, mask)
        # pre-softmax linear shares weights with the embedding
        return y @ self.embed.weight.T

    def forward(self, src, tgt_in):
        return self.decode(tgt_in, self.encode(src))


def lrate(step, d_model, warmup_steps):
    step = max(step, 1)
    return d_model ** -0.5 * min(step ** -0.5, step * warmup_steps ** -1.5)


def make_batch(n):
    src = torch.randint(0, n_symbols, (n, seq_len), device=device)
    tgt = src.flip(1)
    # decoder input is the target shifted right by one
    tgt_in = torch.cat([torch.full((n, 1), bos, device=device), tgt[:, :-1]], dim=1)
    return src, tgt_in, tgt


@torch.no_grad()
def greedy_decode(model, src):
    memory = model.encode(src)
    out = torch.full((src.size(0), 1), bos, device=device)
    for _ in range(seq_len):
        next_token = model.decode(out, memory)[:, -1].argmax(-1, keepdim=True)
        out = torch.cat([out, next_token], dim=1)
    return out[:, 1:]


def train(model, steps, warmup_steps, batch=64, eps_ls=0.1):
    opt = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
    # lr=1.0 times the lambda gives exactly Eq. 3
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lrate(s + 1, model.d_model, warmup_steps))
    curve = []
    for _ in range(steps):
        src, tgt_in, tgt = make_batch(batch)
        logits = model(src, tgt_in)
        loss = F.cross_entropy(logits.reshape(-1, vocab), tgt.reshape(-1), label_smoothing=eps_ls)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        curve.append(loss.item())
    return curve


def test():
    torch.manual_seed(1)
    b, t, d_model, h = 3, 5, 32, 4
    q, k, v = (torch.randn(b, t, d_model, device=device) for _ in range(3))

    out, w = attention(q, k, v)
    assert out.shape == (b, t, d_model) and w.shape == (b, t, t)
    assert torch.allclose(w.sum(-1), torch.ones(b, t, device=device))

    mha = MultiHeadAttention(d_model, h).to(device)
    assert mha(q, k, v).shape == (b, t, d_model)

    # reference: Concat(head_1..head_h) W^O with an explicit loop over heads
    d_k = d_model // h
    heads = []
    for i in range(h):
        cols = slice(i * d_k, (i + 1) * d_k)
        w_q, w_k, w_v = (m.weight[cols].T for m in (mha.w_q, mha.w_k, mha.w_v))
        heads.append(attention(q @ w_q, k @ w_k, v @ w_v)[0])
    naive = torch.cat(heads, dim=-1) @ mha.w_o.weight.T
    assert torch.allclose(mha(q, k, v), naive, atol=1e-5)

    pe = positional_encoding(10, 16)
    assert torch.isclose(pe[3, 4], torch.tensor(math.sin(3 / 10000 ** (4 / 16))))
    assert torch.isclose(pe[3, 5], torch.tensor(math.cos(3 / 10000 ** (4 / 16))))

    model = Transformer(vocab).to(device).eval()
    src, tgt_in, tgt = make_batch(4)
    logits = model(src, tgt_in)
    assert logits.shape == (4, seq_len, vocab)

    # changing future decoder inputs must not change earlier outputs
    changed = tgt_in.clone()
    changed[:, 5:] = (changed[:, 5:] + 1) % n_symbols
    assert torch.allclose(model(src, changed)[:, :5], logits[:, :5], atol=1e-5)
    assert not torch.allclose(model(src, changed)[:, 5:], logits[:, 5:])

    model = Transformer(vocab, p_drop=0.0).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    src, tgt_in, tgt = make_batch(8)
    for _ in range(200):
        loss = F.cross_entropy(model(src, tgt_in).reshape(-1, vocab), tgt.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.05, loss.item()
    print(f"tests passed (overfit loss {loss.item():.1e})")


def softmax_saturation(d_k, n=2000):
    q, k = torch.randn(n, 10, d_k), torch.randn(n, 10, d_k)
    scores = q @ k.transpose(-2, -1)
    unscaled = scores.softmax(-1).amax(-1).mean().item()
    scaled = (scores / math.sqrt(d_k)).softmax(-1).amax(-1).mean().item()
    return unscaled, scaled


def reversal_attention(model, src, tgt_in):
    # mass that target position i puts on source position T-1-i, per layer and head
    model.eval()
    with torch.no_grad():
        model(src, tgt_in)
    t = src.size(1)
    mirror = torch.arange(t - 1, -1, -1, device=src.device)
    rows = torch.arange(t, device=src.device)
    return torch.stack([layer.cross_attn.weights[:, :, rows, mirror].mean((0, 2)) for layer in model.decoder])


def accuracy(model, n=1000):
    model.eval()
    src, _, tgt = make_batch(n)
    pred = greedy_decode(model, src)
    return (pred == tgt).float().mean().item(), (pred == tgt).all(1).float().mean().item()


def ablation(change, steps, warmup_steps):
    torch.manual_seed(0)
    model = Transformer(vocab).to(device)
    if change == "no positional encoding":
        model.use_pe = False
    if change == "no 1/sqrt(d_k)":
        for m in model.modules():
            if isinstance(m, MultiHeadAttention):
                m.scale = False
    if change == "no warmup":
        warmup_steps = 1
    curve = train(model, steps, warmup_steps)
    return accuracy(model), curve[-1]


def plot(curve, attn, path):
    try:
        import matplotlib
    except ImportError:
        print("matplotlib not installed, skipping the plot")
        return
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink = "#8a8a86"
    window = 25
    smooth = [sum(curve[max(0, i - window):i + 1]) / len(curve[max(0, i - window):i + 1]) for i in range(len(curve))]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.8), dpi=150, gridspec_kw={"width_ratios": [1.4, 1]})
    ax1.plot(smooth, color="#2a78d6", lw=2)
    ax1.axhline(math.log(n_symbols), color=ink, lw=1, ls="--")
    ax1.annotate("ln 10, uniform guess", (len(curve), math.log(n_symbols)), xytext=(-4, 4),
                 textcoords="offset points", ha="right", color=ink, fontsize=8)
    ax1.set_title("training loss (label smoothing 0.1)", color=ink, fontsize=10)
    ax1.set_xlabel("step", color=ink)
    ax1.set_ylabel("cross-entropy", color=ink)
    ax1.grid(axis="y", color=ink, alpha=0.2)
    for side in ("top", "right"):
        ax1.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax1.spines[side].set_color(ink)

    im = ax2.imshow(attn.cpu(), cmap="Blues", vmin=0, vmax=1)
    ax2.set_title("cross-attention, last decoder layer", color=ink, fontsize=10)
    ax2.set_xlabel("source position", color=ink)
    ax2.set_ylabel("target position", color=ink)
    cbar = fig.colorbar(im, ax=ax2, fraction=0.046)
    cbar.ax.tick_params(colors=ink)
    cbar.outline.set_edgecolor(ink)
    for spine in ax2.spines.values():
        spine.set_color(ink)
    for ax in (ax1, ax2):
        ax.tick_params(which="both", colors=ink)
    fig.suptitle("Transformer learning to reverse sequences", color=ink)
    fig.tight_layout()
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path, transparent=True)
    print(f"saved {path.name}")


if __name__ == "__main__":
    test()
    if "--test" in sys.argv:
        sys.exit()
    start = time.time()

    for d_k in (4, 16, 64):
        unscaled, scaled = softmax_saturation(d_k)
        print(f"d_k={d_k:<3} mean max attention weight: unscaled {unscaled:.2f}, scaled {scaled:.2f}")

    torch.manual_seed(0)
    model = Transformer(vocab).to(device)
    warmup_steps, steps = 200, 1000
    curve = train(model, steps, warmup_steps)
    print(f"params {sum(p.numel() for p in model.parameters())}, "
          f"loss at step 1 {curve[0]:.2f} (ln {vocab} = {math.log(vocab):.2f}), final train loss {curve[-1]:.3f}")

    token_acc, seq_acc = accuracy(model)
    print(f"test: token accuracy {token_acc:.3f}, sequence accuracy {seq_acc:.3f}")
    src, tgt_in, _ = make_batch(1)
    print(f"example: {src[0].tolist()} -> {greedy_decode(model, src)[0].tolist()}")

    src, tgt_in, _ = make_batch(500)
    mass = reversal_attention(model, src, tgt_in)
    for layer, heads in enumerate(mass):
        print(f"decoder layer {layer + 1} cross-attention on the mirrored source position, per head: "
              + " ".join(f"{m:.2f}" for m in heads.tolist()) + f" (uniform = {1 / seq_len:.2f})")
    with torch.no_grad():
        model(src[:1], tgt_in[:1])
    last = model.decoder[-1].cross_attn.weights[0]
    best_head = mass[-1].argmax()
    print(f"total time {time.time() - start:.1f}s")
    plot(curve, last[best_head], Path(__file__).parent / "assets" / "results.png")

    if "--ablate" in sys.argv:
        print(f"\nablations, {steps} steps each, same seed")
        print(f"{'change':<24}{'token acc':>10}{'seq acc':>9}{'final loss':>12}")
        print(f"{'none':<24}{token_acc:>10.3f}{seq_acc:>9.3f}{curve[-1]:>12.3f}")
        for change in ("no positional encoding", "no 1/sqrt(d_k)", "no warmup"):
            (tok, seq), loss = ablation(change, steps, warmup_steps)
            print(f"{change:<24}{tok:>10.3f}{seq:>9.3f}{loss:>12.3f}")
