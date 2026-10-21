import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
device = "cpu"

n_prompts, n_responses = 16, 8
n_pairs = 20000
beta = 0.5
steps, lr = 2000, 0.05
betas_frontier = (4.0, 2.0, 1.0, 0.5, 0.25)


class TabularPolicy(nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.logits = nn.Parameter(logits.clone())

    def log_probs(self):
        return self.logits.log_softmax(-1)

    def forward(self, x, y):
        return self.log_probs()[x, y]


def make_world(seed=0):
    g = torch.Generator().manual_seed(seed)
    ref_logits = torch.randn(n_prompts, n_responses, generator=g)
    r_star = torch.randn(n_prompts, n_responses, generator=g)
    return ref_logits, r_star


def optimal_policy(ref_log_probs, r, beta):
    # eq. 4: pi*(y|x) = pi_ref(y|x) exp(r(x,y) / beta) / Z(x)
    return (ref_log_probs + r / beta).log_softmax(-1)


def make_preferences(ref_log_probs, r_star, n, seed=1):
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(0, n_prompts, (n,), generator=g)
    probs = ref_log_probs.exp()[x]
    y1 = torch.multinomial(probs, 1, generator=g).squeeze(1)
    # second sample from pi_ref, conditioned on being different from the first
    y2 = torch.multinomial(probs.scatter(1, y1[:, None], 0.0), 1, generator=g).squeeze(1)
    # eq. 1: bradley-terry, p(y1 > y2) = sigmoid(r*(x,y1) - r*(x,y2))
    p1 = torch.sigmoid(r_star[x, y1] - r_star[x, y2])
    first_wins = torch.rand(n, generator=g) < p1
    y_w = torch.where(first_wins, y1, y2)
    y_l = torch.where(first_wins, y2, y1)
    return x, y_w, y_l


def dpo_loss(pi_logp_w, pi_logp_l, ref_logp_w, ref_logp_l, beta):
    # eq. 7
    logits = beta * ((pi_logp_w - ref_logp_w) - (pi_logp_l - ref_logp_l))
    return -F.logsigmoid(logits).mean()


def unweighted_loss(pi_logp_w, pi_logp_l, ref_logp_w, ref_logp_l, beta):
    # the same gradient direction without the sigmoid weight (paper, appendix table 3)
    return -(beta * ((pi_logp_w - ref_logp_w) - (pi_logp_l - ref_logp_l))).mean()


def train_policy(loss_name, ref_logits, data, beta, steps=steps, lr=lr):
    x, y_w, y_l = data
    policy = TabularPolicy(ref_logits)
    ref_lp = ref_logits.log_softmax(-1)
    ref_w, ref_l = ref_lp[x, y_w], ref_lp[x, y_l]
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    for _ in range(steps):
        if loss_name == "sft":
            loss = -policy(x, y_w).mean()
        else:
            fn = dpo_loss if loss_name == "dpo" else unweighted_loss
            loss = fn(policy(x, y_w), policy(x, y_l), ref_w, ref_l, beta)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return policy.log_probs().detach()


def train_reward_model(data, steps=steps, lr=lr):
    x, y_w, y_l = data
    r = torch.zeros(n_prompts, n_responses, requires_grad=True)
    opt = torch.optim.Adam([r], lr=lr)
    for _ in range(steps):
        # eq. 2
        loss = -F.logsigmoid(r[x, y_w] - r[x, y_l]).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return r.detach()


def kl(log_p, log_q):
    return (log_p.exp() * (log_p - log_q)).sum(-1).mean().item()


def expected_reward(log_p, r_star):
    return (log_p.exp() * r_star).sum(-1).mean().item()


def centered(r):
    return r - r.mean(-1, keepdim=True)


def test():
    ref_logits, r_star = make_world(seed=5)
    ref_lp = ref_logits.log_softmax(-1)
    data = make_preferences(ref_lp, r_star, 64, seed=6)
    x, y_w, y_l = data
    assert x.shape == y_w.shape == y_l.shape == (64,)
    assert (y_w != y_l).all()

    # pi_theta = pi_ref at init, so the loss is exactly ln 2
    policy = TabularPolicy(ref_logits)
    loss = dpo_loss(policy(x, y_w), policy(x, y_l), ref_lp[x, y_w], ref_lp[x, y_l], beta)
    assert math.isclose(loss.item(), math.log(2), rel_tol=1e-6)

    # autograd matches the paper's gradient; for a tabular policy the softmax terms cancel
    policy.logits.data += 0.3 * torch.randn_like(ref_logits)
    lp = policy.log_probs()
    r_hat = beta * (lp - ref_lp).detach()
    loss = dpo_loss(policy(x, y_w), policy(x, y_l), ref_lp[x, y_w], ref_lp[x, y_l], beta)
    loss.backward()
    weight = torch.sigmoid(r_hat[x, y_l] - r_hat[x, y_w])
    manual = torch.zeros_like(ref_logits)
    manual.index_put_((x, y_w), weight, accumulate=True)
    manual.index_put_((x, y_l), -weight, accumulate=True)
    manual *= -beta / len(x)
    assert torch.allclose(policy.logits.grad, manual, atol=1e-6)

    # lemmas 1 and 2: r + f(x) gives the same preferences and the same optimal policy
    shift = torch.randn(n_prompts, 1)
    p = torch.sigmoid(r_star[x, y_w] - r_star[x, y_l])
    p_shifted = torch.sigmoid((r_star + shift)[x, y_w] - (r_star + shift)[x, y_l])
    assert torch.allclose(p, p_shifted, atol=1e-6)
    assert torch.allclose(optimal_policy(ref_lp, r_star, beta), optimal_policy(ref_lp, r_star + shift, beta), atol=1e-6)

    # eq. 5 recovers r from pi* up to beta log Z(x), and pi* maximises eq. 3
    pi_star = optimal_policy(ref_lp, r_star, beta)
    log_z = torch.logsumexp(ref_lp + r_star / beta, -1, keepdim=True)
    assert torch.allclose(beta * (pi_star - ref_lp) + beta * log_z, r_star, atol=1e-5)

    def objective(log_pi):
        return expected_reward(log_pi, r_star) - beta * kl(log_pi, ref_lp)

    assert math.isclose(objective(pi_star), (beta * log_z).mean().item(), rel_tol=1e-5)
    for _ in range(20):
        other = (pi_star + 0.5 * torch.randn_like(pi_star)).log_softmax(-1)
        assert objective(other) < objective(pi_star)

    tiny = tuple(t[:4] for t in data)
    overfit = TabularPolicy(ref_logits)
    opt = torch.optim.Adam(overfit.parameters(), lr=0.1)
    for _ in range(300):
        tx, tw, tl = tiny
        loss = dpo_loss(overfit(tx, tw), overfit(tx, tl), ref_lp[tx, tw], ref_lp[tx, tl], beta)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.05, loss.item()
    print(f"tests passed (overfit loss {loss.item():.1e})")


def plot(frontier, points, r_hat, r_star, path):
    try:
        import matplotlib
    except ImportError:
        print("matplotlib not installed, skipping the plot")
        return
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink = "#8a8a86"
    blue, orange, green = "#2a78d6", "#eb6834", "#1baf7a"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.9), dpi=150)

    kls, rewards = zip(*frontier, strict=True)
    ax1.plot(kls, rewards, color=ink, lw=1.5, ls="--", label="optimal π*, eq. 4")
    for key, color, marker, label in (("dpo", blue, "o", "DPO"), ("rm", green, "x", "reward model + exact RL")):
        k, r = zip(*points[key], strict=True)
        ax1.scatter(k, r, color=color, marker=marker, s=40, zorder=3, label=label)
    k, r = points["sft"]
    ax1.scatter([k], [r], color=orange, marker="s", s=40, zorder=3, label="SFT on preferred")
    k, r = points["unweighted"]
    ax1.scatter([k], [r], color=orange, marker="v", s=40, zorder=3, label=f"DPO without σ weight, β={beta}")
    ax1.set_xlabel("KL(π ‖ π_ref)", color=ink)
    ax1.set_ylabel("expected true reward", color=ink)
    ax1.set_title("reward vs KL, one point per β", color=ink, fontsize=10)
    ax1.legend(frameon=False, labelcolor=ink, fontsize=8, loc="lower right")

    ax2.scatter(centered(r_star).flatten(), centered(r_hat).flatten(), s=10, color=blue, alpha=0.7)
    lim = centered(r_star).abs().max().item() * 1.1
    ax2.plot([-lim, lim], [-lim, lim], color=ink, lw=1, ls="--")
    ax2.set_xlabel("true reward r*(x, y), centred per prompt", color=ink)
    ax2.set_ylabel("β log π_θ / π_ref, centred", color=ink)
    ax2.set_title(f"the implicit reward of the DPO policy (β={beta})", color=ink, fontsize=10)

    for ax in (ax1, ax2):
        ax.tick_params(which="both", colors=ink)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(ink)
        ax.grid(color=ink, alpha=0.2)
    fig.tight_layout()
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path, transparent=True)
    print(f"saved {path.name}")


if __name__ == "__main__":
    test()
    if "--test" in sys.argv:
        sys.exit()
    start = time.time()

    ref_logits, r_star = make_world()
    ref_lp = ref_logits.log_softmax(-1)
    data = make_preferences(ref_lp, r_star, n_pairs)
    pi_star = optimal_policy(ref_lp, r_star, beta)
    print(f"{n_prompts} prompts, {n_responses} responses, {n_pairs} preference pairs sampled from π_ref, β={beta}")

    reward_model = train_reward_model(data)
    policies = {
        "π_ref": ref_lp,
        "SFT on preferred": train_policy("sft", ref_logits, data, beta),
        "DPO, no σ weight": train_policy("unweighted", ref_logits, data, beta),
        "reward model + exact RL": optimal_policy(ref_lp, reward_model, beta),
        "DPO": train_policy("dpo", ref_logits, data, beta),
        "optimal π*": pi_star,
    }
    print(f"\n{'policy':<26}{'KL(π‖π*)':>10}{'KL(π‖π_ref)':>13}{'E[r*]':>8}{'min log π':>11}")
    for key, log_pi in policies.items():
        print(f"{key:<26}{kl(log_pi, pi_star):>10.4f}{kl(log_pi, ref_lp):>13.3f}"
              f"{expected_reward(log_pi, r_star):>8.3f}{log_pi.min():>11.1f}")

    dpo, rm = policies["DPO"], policies["reward model + exact RL"]
    print(f"\nmax |log π_DPO - log π_RM+RL| = {(dpo - rm).abs().max():.1e}")
    r_hat = beta * (dpo - ref_lp)
    err = centered(r_hat) - centered(r_star)
    corr = torch.corrcoef(torch.stack([centered(r_hat).flatten(), centered(r_star).flatten()]))[0, 1]
    rmse = err.pow(2).mean().sqrt()
    print(f"implicit reward vs r*, both centred per prompt: correlation {corr:.3f}, rmse {rmse:.3f}")
    rare = ref_lp.exp() < 0.05
    print(f"rmse on responses with π_ref < 0.05: {err[rare].pow(2).mean().sqrt():.3f}, "
          f"on the rest: {err[~rare].pow(2).mean().sqrt():.3f}")

    frontier, points = [], {"dpo": [], "rm": []}
    for b in torch.logspace(math.log10(8), math.log10(0.1), 40).tolist():
        p = optimal_policy(ref_lp, r_star, b)
        frontier.append((kl(p, ref_lp), expected_reward(p, r_star)))
    for b in betas_frontier:
        p = train_policy("dpo", ref_logits, data, b)
        points["dpo"].append((kl(p, ref_lp), expected_reward(p, r_star)))
        p = optimal_policy(ref_lp, reward_model, b)
        points["rm"].append((kl(p, ref_lp), expected_reward(p, r_star)))
    sft = policies["SFT on preferred"]
    points["sft"] = (kl(sft, ref_lp), expected_reward(sft, r_star))
    unweighted = policies["DPO, no σ weight"]
    points["unweighted"] = (kl(unweighted, ref_lp), expected_reward(unweighted, r_star))
    print(f"total time {time.time() - start:.1f}s")
    plot(frontier, points, r_hat, r_star, Path(__file__).parent / "assets" / "results.png")
