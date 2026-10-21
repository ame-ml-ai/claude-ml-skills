# Test report: paper-to-code on two papers

Environment: Windows 11, CPU only, Python 3.14.0, PyTorch 2.14.1+cpu, matplotlib installed.
Each `main.py` was copied alone into an empty folder and run with `python -I main.py` and with `--test`. Wall times include Python start-up and the tests. They vary with machine load.

## LoRA (arXiv 2106.09685) — `examples/lora`

| Check | Result |
|---|---|
| Runs from scratch on CPU, no downloads | ✅ 16–18 s wall. Synthetic data only. Still runs, without the plot, if matplotlib is missing |
| Shapes test | ✅ shapes of A, B, ΔW and the model output |
| Overfit one batch | ✅ MSE 1.4e-13 on 16 samples in 300 steps |
| Method-specific properties | ✅ output equals the base model at init (B = 0); only A and B are trainable and W0 is bit-identical after training; rank(ΔW) ≤ r; merged weights give the same output (atol 1e-5) |
| Equations match the paper | ✅ forward pass: §4.1 Eq. 3 plus the α/r scaling from §4.1 text (the paper's equation omits it; this is noted in the breakdown); init A ~ Gaussian, B = 0: §4.1; merge W = W0 + BA: §4.1 "No Additional Inference Latency"; parameter count: §5.1 |
| Comments | ✅ 7 comments in ~215 lines, all lowercase, no docstrings |
| Nothing invented | ✅ assumptions marked: σ of A, α = 8, learning rates, where LoRA is placed (MLP layers instead of Wq/Wv). The breakdown states it is not a reproduction |
| Breakdown structure | ✅ header, TL;DR, Problem, Core idea, Key equations, Step by step, Implementation, Simplifications, Gotchas, Check yourself (answers in `<details>`) |

Toy result: LoRA r=4 closes 90% of the gap between "no adaptation" (MSE 0.388) and full fine-tuning (0.059), with 13× fewer trainable parameters. r=1 is clearly too small for a rank-4 shift.

## Attention Is All You Need (arXiv 1706.03762) — `examples/transformer`

| Check | Result |
|---|---|
| Runs from scratch on CPU, no downloads | ✅ 40–51 s wall. Synthetic sequence-reversal task |
| Shapes test | ✅ attention output and weights, multi-head output, model logits |
| Overfit one batch | ✅ cross-entropy 4.6e-3 on 8 samples in 200 steps (dropout and label smoothing off) |
| Method-specific properties | ✅ attention rows sum to 1; vectorised multi-head equals an explicit per-head Concat(...)W^O; PE matches the sin/cos formula; causal mask: changing future decoder inputs leaves earlier outputs unchanged |
| Equations match the paper | ✅ attention: §3.2.1 Eq. 1; multi-head: §3.2.2; FFN: §3.3 Eq. 2; positional encoding: §3.5; post-LN residual: §3.1; dropout placement: §5.4; masking with −∞: §3.2.3; shared embeddings and the √d_model factor: §3.4; Adam betas/eps and lrate: §5.3 Eq. 3; label smoothing: §5.4 |
| Comments | ✅ 10 comments in ~300 lines, all lowercase, no docstrings |
| Nothing invented | ✅ assumptions marked: embedding init, step numbering from 1, warmup 200, greedy decoding instead of beam search. The breakdown states it is not a reproduction of BLEU |
| Breakdown structure | ✅ same structure as above |

Toy result: 100% token and sequence accuracy on 1000 held-out sequences after 1000 steps. Without the 1/√d_k factor, the mean max attention weight goes 0.49 → 0.86 as d_k goes 4 → 64. With it, the weight stays at 0.32.

## What failed during testing, and the fix in SKILL.md

| # | Problem found | Root cause in SKILL.md | Fix |
|---|---|---|---|
| 1 | First LoRA run was meaningless: targets had variance 0.004, so all methods looked alike | Nothing asked for the toy setting to be informative | New subsection "Make the toy setting informative": unit-variance targets, a planted structure, and a setting that should fail |
| 2 | Transformer started at loss 46 (should be ≈ ln 11 = 2.4) and reached only 78% sequence accuracy. Cause: default N(0,1) embeddings × √d_model with tied output weights | No sanity check on the initial loss; no rule for unspecified initialization | Check the step-0 loss; choose a unit-scale init when the paper is silent and mark it as an assumption; record such bugs under Gotchas with their symptom |
| 3 | Comments `#A gaussian…`, `#B = 0…` broke the lowercase rule | The rule did not cover capitalised math symbols | Rule now says how to rephrase |
| 4 | `results.png` needs matplotlib, but Step 4 allowed only torch/numpy | Step 4 and Step 6 contradicted each other | Matplotlib is optional and lazily imported; the plot is skipped if it is missing |
| 5 | Minor tick labels on the log axis stayed black (invisible on dark GitHub) | "Readable on light and dark" gave no recipe | Concrete plot recipe: transparent background, one mid-gray ink, `which="both"`, ≤ 4 series, no dual axes, look at the PNG |
| 6 | Transformer run took 76–96 s at first | Unclear whether "under a minute" includes tests | The whole `python main.py`, tests included, counts; measure it; shrink the model, not the idea |
| 7 | The Transformer HTML fetch was truncated (appendix unread) | Nothing about partial reads or versions | Fetch the rest, name unread parts, and record the arXiv version in a header block |
| 8 | Unclear how `test()` and `--test` relate; unclear where files go | Ambiguous wording; no file layout | `test()` always runs first, `--test` runs tests only; new "Saving to files" layout; re-run before copying output; check every number in the prose |
