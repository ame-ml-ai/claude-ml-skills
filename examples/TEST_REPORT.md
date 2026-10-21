# How the examples were checked

Each example was generated with the `paper-to-code` skill, then run, read line by line against the paper, and revised. This file records what was checked and, more usefully, what went wrong on the way.

## Environment

Linux, 2 CPU cores, Python 3.13, PyTorch 2.14.1 (CPU). The first versions of LoRA and Transformer were also run on Windows 11 with Python 3.14. CI runs every example's tests and full demo on each push (Ubuntu, Python 3.12, CPU-only PyTorch), see [`.github/workflows/examples.yml`](../.github/workflows/examples.yml).

| Example | `python main.py`, wall time | Tests |
|---|---|---|
| [LoRA](lora) | 39 s | shapes, overfit one batch, output equals the base model at init, only A and B train and W0 stays bit-identical, rank(ΔW) = r, merged weights give the same output, subspace similarity is 1 for the same span and 0 for orthogonal spans |
| [Transformer](transformer) | 35 s (2 min 15 s with `--ablate`) | shapes, overfit one batch, attention rows sum to 1, vectorised multi-head equals an explicit per-head loop, PE matches the formula, changing future decoder inputs leaves earlier outputs unchanged |
| [DPO](dpo) | 25 s | loss is exactly ln 2 at init, autograd equals the paper's gradient formula, r + f(x) leaves preferences and the optimal policy unchanged, Eq. 5 recovers r, Eq. 4 maximises Eq. 3, overfit 4 pairs |

Every number quoted in a breakdown was copied from the final run and checked against it.

## What the toy experiments are designed to show

A demo that only shows "the loss went down" proves little. Each example is built so that a specific claim of the paper can succeed or fail:

- **LoRA.** The pretrained model is the task-A network itself, and task B adds a known update ΔW*. So we can check whether LoRA recovers ΔW* (it does: 3.6% relative error, subspace similarity 1.000 at r = 4), not only whether the loss is low. A full-rank shift with the same norm is the negative control: there LoRA r = 4 is 5.6× worse than full fine-tuning. Learning rates are tuned per method on a validation split; results are mean ± std over 3 seeds.
- **Transformer.** Sequence reversal has a known correct alignment (target i reads source T-1-i). The last decoder layer's cross-attention puts 0.99–1.00 of its weight there. Three ablations remove one ingredient each.
- **DPO.** In a contextual bandit with 8 responses, the KL-regularised optimum (Eq. 4) can be computed exactly. DPO lands on that optimal reward/KL frontier for every β tried, and gives the same policy as "fit a reward model, then solve the RL step exactly", which is the paper's central equivalence.

## What went wrong, and what changed in the skill

| # | Example | What happened | Fix in `SKILL.md` |
|---|---|---|---|
| 1 | LoRA | The first run was uninformative: targets had variance 0.004, so every method looked the same. | Scale synthetic targets to unit variance; plant a structure the method should find and a setting where it should fail. |
| 2 | Transformer | Initial loss 46 instead of ln 11 ≈ 2.4, and 78% sequence accuracy. Cause: PyTorch's default N(0, 1) embeddings times √d_model, with tied output weights. | Check the step-0 loss. When the paper does not specify an init, use a unit-scale one and mark it as an assumption. Record the bug under Gotchas with its symptom. |
| 3 | LoRA | v1 compared methods at hand-picked learning rates on one seed, and its student was trained from scratch, so the "true" update was unknown and LoRA's result could not be checked beyond the loss. | New "Compare fairly" section: per-method learning rate on a validation split, mean ± std over seeds. Prefer toys with a known answer and measure the distance to it. Add a negative control. |
| 4 | DPO | Expected the version without the σ weight to blow up the KL. It did not: with Adam, every net-winning response grows at the same rate, so their ratios to π_ref stay fixed, while net losers go to log π = −∞ (−204.8 after 2000 steps). The policy still ends up below the optimal frontier. | The breakdown reports what happened, not what was expected. |
| 5 | Transformer | Removing 1/√d_k did not hurt at all (d_k = 16). | Report negative ablation results; explain them (the argument is about large d_k). |
| 6 | LoRA | The paper defines the subspace similarity with right singular vectors in §7.2 and left singular vectors in Appendix G. | Name inconsistencies in the paper under Gotchas and say which reading the code uses. |
| 7 | all | Comments were written as `#comment`, which PEP 8 and linters reject. | Comments are `# comment`; CI runs `ruff`. |
| 8 | LoRA, Transformer | `results.png` needed matplotlib, but the scope rule allowed only torch and numpy. | Matplotlib is optional and imported lazily; the plot is skipped if it is missing. |
| 9 | LoRA | Minor tick labels on a log axis stayed black, invisible on GitHub's dark theme. | Concrete plot recipe: transparent background, one mid-gray ink, `tick_params(which="both")`, at most 4 series, look at the PNG. |
| 10 | Transformer | The first version took 76–96 s. | The whole `python main.py`, tests included, must fit in about a minute. Shrink the model, not the idea; slow extras go behind a flag. |
| 11 | Transformer, LoRA | Long papers came back truncated (the LoRA §7 and appendices, the Transformer appendix). | Fetch the rest, name any unread parts in the breakdown header, and record the arXiv version read. |
