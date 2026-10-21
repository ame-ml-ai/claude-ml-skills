**English** | [Русский](README.ru.md)

# claude-ml-skills

Claude skills for ML work. The first one, `paper-to-code`, turns a paper into a breakdown you can learn from and a small PyTorch implementation whose claims are tested, not just plotted.

[![examples](https://github.com/ame-ml-ai/claude-ml-skills/actions/workflows/examples.yml/badge.svg)](https://github.com/ame-ml-ai/claude-ml-skills/actions/workflows/examples.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Claude skill](https://img.shields.io/badge/Claude-skill-D97757.svg)](skills/paper-to-code/SKILL.md)
[![PyTorch](https://img.shields.io/badge/examples-PyTorch-EE4C2C.svg)](examples)

<p align="center">
  <img src="examples/dpo/assets/results.png" width="90%" alt="DPO policies land on the exact reward/KL frontier; the implicit reward matches the true reward">
  <br><sub>DPO on a toy bandit where the KL-regularised optimum is known exactly. Left: for every β, the DPO policy lands on the optimal reward/KL frontier and coincides with "reward model + exact RL". Right: β log π/π_ref recovers the true reward up to a per-prompt constant.</sub>
</p>

## Examples

All three were generated with the skill, run, and checked against the papers. Breakdowns are in English and Russian.

| Paper | What the toy run shows | |
|---|---|---|
| [**LoRA**](examples/lora) · [2106.09685](https://arxiv.org/abs/2106.09685) | On a planted rank-4 shift, LoRA r=4 recovers the true update to 3.6% and beats full fine-tuning (MSE 0.0007 vs 0.0234, 13× fewer parameters). On a full-rank shift of the same norm it loses 5.6×. Per-method learning rates, 3 seeds. | [breakdown](examples/lora/breakdown.md) · [code](examples/lora/main.py) |
| [**Transformer**](examples/transformer) · [1706.03762](https://arxiv.org/abs/1706.03762) | Learns sequence reversal (999/1000 exact) and the correct anti-diagonal cross-attention. Ablations: without positional encoding accuracy drops to the bag-of-tokens ceiling (0.21); without warmup training collapses; without 1/√d_k nothing changes at d_k = 16. | [breakdown](examples/transformer/breakdown.md) · [code](examples/transformer/main.py) |
| [**DPO**](examples/dpo) · [2305.18290](https://arxiv.org/abs/2305.18290) | Matches "reward model + exact RL" to 4 decimals and lands on the optimal frontier; the implicit reward correlates 0.976 with the true one. Without the σ weight, losing responses go to log π = −∞. | [breakdown](examples/dpo/breakdown.md) · [code](examples/dpo/main.py) |

<p align="center">
  <img src="examples/lora/assets/results.png" width="49%" alt="LoRA: test MSE on rank-4 and full-rank shifts; singular values of the learned update vs the planted one">
  <img src="examples/transformer/assets/results.png" width="49%" alt="Transformer: training loss with a plateau at ln 10; anti-diagonal cross-attention">
</p>

Each example is one `main.py` that runs on a laptop CPU in under a minute, downloads nothing, and starts with its own tests:

```bash
cd examples/dpo
pip install -r requirements.txt
python main.py           # tests, experiment, assets/results.png
python main.py --test    # tests only
```

[`examples/TEST_REPORT.md`](examples/TEST_REPORT.md) lists what was checked and what went wrong while building them (an initial loss of 46 instead of 2.4, an ablation that changed nothing, a paper that defines the same quantity in two different ways).

## What the skill does

Give Claude an arXiv link, ID or PDF. It returns:

1. **A breakdown:** TL;DR, problem, core idea, the key equations with tensor shapes and section numbers mapped to functions in the code, a step-by-step walkthrough, simplifications, gotchas, and three self-check questions.
2. **A minimal implementation:** one `main.py` with tests (shapes, overfitting one batch, a property specific to the method) and a demo that prints real numbers and saves a plot.

The rules it follows:

- Read the actual paper, appendices included, and say which version was read and what could not be read. Never reconstruct a paper from memory.
- Mark every value the paper does not specify as an assumption.
- Build the toy so that the paper's claim can succeed or fail: a planted structure or a closed-form optimum to measure against, and a negative control where the method should lose.
- Compare fairly: each method gets its own learning rate chosen on a validation split; close results get several seeds.
- Check the loss at step 0 against what it should be (ln C for classification, ln 2 for DPO).
- Run the code and copy the output from the final run. Never claim that a toy run reproduces the paper's results.

It answers in the language you write in; code stays in English. The full instructions are in [`SKILL.md`](skills/paper-to-code/SKILL.md).

## Install

**claude.ai**

1. Download [`skills/paper-to-code`](skills/paper-to-code) and zip the `paper-to-code` folder.
2. Go to **Settings → Capabilities → Skills**, choose **Upload skill**, and select the zip.

**Claude Code**

```bash
git clone https://github.com/ame-ml-ai/claude-ml-skills.git
cp -r claude-ml-skills/skills/paper-to-code ~/.claude/skills/
```

Restart Claude Code. The skill loads when you share a paper.

## Usage

```text
Here is a paper: https://arxiv.org/abs/2305.18290
Explain it and implement the core idea in PyTorch.
```

- `quick` gives only the TL;DR, core idea and equations.
- `code only` gives just the implementation.
- "publish to GitHub" prepares a repo with a README, results plot and citation, after confirming the name and visibility.

## Roadmap

- **pytorch-debugger**: find shape errors, NaNs and silent training bugs step by step.
- **eval-builder**: design and write evaluation suites for ML models.
- **kazakh-ml-glossary**: consistent Kazakh translations of ML terms.

## Contributing

Issues and pull requests are welcome, especially bug reports from real papers where the skill got something wrong. To add an example, follow the layout in [`examples/`](examples), include the output of the final run, and make sure `python main.py --test` passes in CI.

## License

[MIT](LICENSE). The breakdowns describe the original papers; credit belongs to their authors, cited in each breakdown.
