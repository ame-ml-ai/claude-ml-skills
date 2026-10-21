**English** | [Русский](breakdown.ru.md)

# LoRA: Low-Rank Adaptation of Large Language Models

**Paper:** Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen (Microsoft). [arXiv:2106.09685](https://arxiv.org/abs/2106.09685), v1 17 Jun 2021. Read: v2, 16 Oct 2021 (PDF and HTML, including §7 and Appendix G).
**Code:** [`main.py`](main.py)

## TL;DR

Full fine-tuning stores a copy of every weight for every task. LoRA freezes the pretrained weights $W_0$ and learns only a low-rank update $\Delta W = BA$, which cuts trainable parameters by orders of magnitude. After training, $BA$ is added into $W_0$, so inference costs nothing extra. On GPT-3 175B it matches or beats full fine-tuning (Table 4).

## Problem

With full fine-tuning the update $\Delta\Phi$ is as large as the model, $|\Delta\Phi| = |\Phi_0|$ (§2): 175B parameters per task for GPT-3. Adapters add sequential layers and inference latency; prefix tuning eats part of the context window and is hard to optimise (§3).

## Core idea

The authors hypothesise that the weight change needed to adapt to a new task has a low *intrinsic rank*: the weights only need to move inside a small subspace. So instead of a full $d \times k$ matrix $\Delta W$, LoRA learns two thin matrices $B$ ($d \times r$) and $A$ ($r \times k$) with $r \ll \min(d, k)$.

**Difference from the baseline:** full fine-tuning updates $W$ itself. LoRA keeps $W_0$ frozen and trains only $B$ and $A$, which sit next to it.

## Key equations

**1. Forward pass** (§4.1, Eq. 3, plus the $\alpha/r$ scale from the text right after it):

$$h = W_0 x + \frac{\alpha}{r} B A x$$

- $x \in \mathbb{R}^{k}$ is the layer input (a batch $x \in \mathbb{R}^{N \times k}$ in the code)
- $W_0 \in \mathbb{R}^{d \times k}$ frozen, $B \in \mathbb{R}^{d \times r}$ and $A \in \mathbb{R}^{r \times k}$ trainable
- $\alpha$ is a constant: "we simply set α to the first r we try and do not tune it"
- code: `LoRALinear.forward`

> Eq. 3 itself has no $\alpha/r$. The sentence after it does: "We then scale ∆Wx by α/r". The code includes it.

**2. Initialisation** (§4.1): "random Gaussian initialization for A and zero for B, so ∆W = BA is zero at the beginning of training".
- code: `LoRALinear.__init__`

**3. Merge for inference** (§4.1, "No Additional Inference Latency"): $W = W_0 + \frac{\alpha}{r} B A$.
- code: `LoRALinear.merge`, `LoRALinear.delta_w`

**4. Normalised subspace similarity** (§7.2, Eq. 4; Appendix G):

$$\phi(A_1, A_2, i, j) = \frac{\lVert U_{A_1}^{i\top} U_{A_2}^{j} \rVert_F^2}{\min(i, j)} \in [0, 1]$$

- $U^i$ holds the top $i$ singular vectors; 1 means one subspace contains the other, 0 means they are orthogonal
- the paper uses it to show that the top directions learned with $r=8$ and $r=64$ overlap while the rest look like noise
- code: `subspace_similarity`, applied to the learned $\Delta W$ and the planted $\Delta W^*$

## How it works step by step

1. Take a pretrained model (here an MLP, `teacher_a`).
2. Freeze all of its parameters (`add_lora`).
3. Wrap the chosen linear layers in `LoRALinear`: next to $W_0$ there is now $A$ (Gaussian) and $B$ (zeros).
4. Forward: $h = W_0 x + \frac{\alpha}{r} B(Ax)$. The input is projected to $r$ dimensions and back. The $d \times k$ matrix $BA$ is never formed during training.
5. Because $B = 0$, the model starts exactly at the pretrained one.
6. Gradients flow only to $A$ and $B$; the optimiser sees only them.
7. After training, `merge()` folds $W_0 + \frac{\alpha}{r}BA$ into a plain `nn.Linear`.

## Implementation

### Setup

The toy is built so that the right answer is known.

- **Pretrained model:** a random MLP $32 \to 128 \to 128 \to 8$ (`teacher_a`), with outputs scaled to unit variance. We skip pretraining and use this network itself as the pretrained model.
- **Task B:** the same network with a planted update $\Delta W^*$ added to `fc1` and `fc2`. Because the pretrained model *is* task A's network, $\Delta W^*$ is exactly the update adaptation needs.
- **Two versions of task B**, same Frobenius norm $\lVert \Delta W^* \rVert_F = 3$: a **rank-4** shift (LoRA's hypothesis holds) and a **full-rank** shift (it does not, a negative control).
- **Adaptation:** 1000 training examples, 600 Adam steps. Full fine-tuning vs LoRA $r \in \{1, 4, 8\}$ on `fc1` and `fc2`.
- **Fair comparison:** every method gets its own learning rate, picked from a 3-value grid on a separate validation set. The test set is used only once. Results are mean ± std over 3 training seeds (init of $A$ and batch order; the data is fixed).

```bash
pip install -r requirements.txt
python main.py          # tests + experiment + assets/results.png
python main.py --test   # tests only
```

Output (Linux, 2 CPU cores, Python 3.13, PyTorch 2.14.1; `python main.py` takes 39 s wall time, tests included):

```
tests passed (overfit loss 3.2e-13)

test MSE, mean ± std over 3 seeds, lr picked on a validation set
method              params      lr      rank-4 shift   full-rank shift
no adaptation            0       -     0.8478 ± 0.0000     0.4968 ± 0.0000
full fine-tuning     21768   1e-03     0.0234 ± 0.0001     0.0217 ± 0.0001
LoRA r=1               416   3e-03     0.3033 ± 0.0033     0.2312 ± 0.0029
LoRA r=4              1664   1e-02     0.0007 ± 0.0002     0.1220 ± 0.0029
LoRA r=8              3328   1e-02     0.0018 ± 0.0003     0.0648 ± 0.0048

rank-4 shift, fc1: does the learned ΔW find the planted ΔW*?
method             rel. error   φ(ΔW, ΔW*)   top singular values
full fine-tuning        0.719        0.966   0.94 0.76 0.67 0.55 0.29 0.27
LoRA r=1                1.196        0.494   2.71 0.00 0.00 0.00 0.00 0.00
LoRA r=4                0.036        1.000   1.98 1.58 1.28 0.92 0.00 0.00
LoRA r=8                0.078        0.999   1.94 1.58 1.27 0.91 0.09 0.08
planted ΔW*                                  1.98 1.59 1.29 0.93 0.00 0.00

total time 33.9s
saved results.png
```

![LoRA vs full fine-tuning](assets/results.png)

### What the numbers show

- **When the shift really is low-rank, LoRA r=4 beats full fine-tuning:** test MSE 0.0007 vs 0.0234, with 13× fewer trainable parameters. With 1000 examples, the rank constraint works as a correct prior. Full fine-tuning can move all 21768 parameters, and from 1000 examples it finds a function that fits the training set but generalises worse (seed 0: train MSE 0.003, test 0.023; the targets have no label noise, so this is a generalisation gap, not noise fitting).
- **When the shift is full-rank, the order flips:** full fine-tuning gets 0.0217, LoRA r=4 gets 0.1220 (5.6× worse) and r=8 gets 0.0648. LoRA is only as good as the low-rank hypothesis. The paper checks that hypothesis empirically for LLMs (§7); it is not guaranteed.
- **LoRA finds the planted update itself, not just a function that fits.** For r=4, $\Delta W$ of `fc1` is within 3.6% of $\Delta W^*$ (relative Frobenius error), with subspace similarity $\phi = 1.000$, and its singular values match $\Delta W^*$ to two decimals.
- **The extra directions for r=8 stay small** (0.09 and 0.08 next to 1.94 … 0.91). This is the toy version of the §7.2 finding: the top directions carry the signal and the rest "contain mostly random noises".
- **Full fine-tuning's `fc1` update points the right way but is too small** ($\phi = 0.966$, singular values about half of $\Delta W^*$, relative error 0.72). It also changes `fc3` and the biases, so it does not need to put the whole change into `fc1`.
- **r=1 is too small for a rank-4 shift** (0.3033), as expected.

This is a toy built so that LoRA's hypothesis holds by construction (plus one control where it fails). It shows the mechanism. It does **not** reproduce the paper's GPT-3, RoBERTa or GPT-2 results.

### Tests (`test()`)

- **shapes:** $A \in \mathbb{R}^{r \times k}$, $B \in \mathbb{R}^{d \times r}$, $\Delta W$ has the shape of $W_0$, the trainable parameter count is $r(d + k)$ per layer;
- **overfit one batch:** LoRA r=4 drives MSE on 16 samples to $3 \times 10^{-13}$ in 300 steps;
- **method properties:**
  - with $B = 0$ the LoRA model's output is bit-identical to the base model's;
  - only `A` and `B` are trainable, and $W_0$ is bit-identical after training;
  - $\operatorname{rank}(\Delta W) = r$ after training;
  - the merged model gives the same output as the LoRA model (atol $10^{-5}$);
  - $\phi$ is 1 for a matrix and a rescaled copy of itself, and 0 for orthogonal subspaces.

## Simplifications

- **Scale and data.** A 22K-parameter MLP on synthetic regression instead of 175B-parameter transformers on NLP benchmarks.
- **No pretraining step.** The pretrained model is the task-A network itself, so the needed update is known exactly. In a real model the "true" $\Delta W$ is unknown, and its low rank is a hypothesis.
- **Where LoRA goes.** The paper adapts the attention matrices $W_q, W_v$ (§4.2, §5.1). An MLP has no attention, so LoRA wraps the two hidden layers. The mechanism (frozen `nn.Linear` plus $BA$) is the same.
- **Variance of $A$.** The paper says only "random Gaussian" ($\mathcal{N}(0, \sigma^2)$ in Figure 1). *Assumption:* $\sigma = 1/\sqrt{k}$.
- **$\alpha$.** *Assumption:* $\alpha = 8$ for every $r$, so $\alpha/r$ changes with $r$ as in the paper.
- **Optimiser.** Adam with a per-method learning rate from a 3-value grid, 600 steps, no schedule. Appendix D of the paper uses AdamW with warmup and linear decay.
- **Subspace similarity on $\Delta W$, not on $A$.** The paper compares the $A$ matrices of two runs. We compare against a known $\Delta W^*$, so we use $\Delta W$. For $r=4$ the two are the same subspace (the row space of $BA$ equals that of $A$ when $B$ has full column rank).
- **Biases** are frozen, as in the paper (§4.2). The third layer is not adapted.

## Gotchas

- **Zero $B$, not $A$** (§4.1). Then the model starts at the pretrained one. If both are zero, the gradient of each is proportional to the other, and training never starts.
- **The $\alpha/r$ factor is easy to lose.** It is not in Eq. 3, only in the text of §4.1. Without it, changing $r$ means re-tuning the learning rate.
- **Never form $BA$ during training.** Compute $B(Ax)$: $O(r(d+k))$ memory instead of $O(dk)$.
- **Freeze everything else:** layers without LoRA, LayerNorm and biases (§4.2). Pass only parameters with `requires_grad=True` to the optimiser, otherwise weight decay can still change frozen weights.
- **Merging breaks mixed-task batches** (§4.2). After `merge()` you cannot batch inputs for different $A, B$ together. Keep the LoRA branch separate in that case.
- **Compare against a tuned baseline.** LoRA and full fine-tuning want different learning rates (here 1e-2 vs 1e-3). With one shared learning rate, one of them is handicapped, and the comparison says more about the learning rate than about the method.
- **The paper is inconsistent about which singular vectors $\phi$ uses.** §7.2 says "right-singular unitary matrices" of $A$; Appendix G says "left singular matrices". For $A \in \mathbb{R}^{r \times k}$ only the right singular vectors (in $\mathbb{R}^k$) give a meaningful subspace; the left ones are $r \times r$ and span all of $\mathbb{R}^r$. The code uses right singular vectors.
- **Notation drift:** §7.2 and Table 6 call the attention output projection $W_c$; the rest of the paper calls it $W_o$.

## Check yourself

1. Why is $B$ initialised to zero rather than both matrices?
2. How many trainable parameters does LoRA add to one $1024 \times 4096$ layer with $r = 8$, and how many times fewer is that than the full layer?
3. In the toy, LoRA r=4 beats full fine-tuning on the rank-4 shift but loses on the full-rank shift with the same norm. What does that tell you about when LoRA should work on a real model?

<details>
<summary>Answers</summary>

1. We need $\Delta W = BA = 0$ at the start, so the model starts at the pretrained one. One zero matrix is enough. If both are zero, the gradient with respect to $A$ is proportional to $B$ and vice versa, so both stay zero forever.
2. $r(d + k) = 8 \cdot (1024 + 4096) = 40\,960$ vs $1024 \cdot 4096 = 4\,194\,304$, about 102× fewer.
3. LoRA restricts the update to rank $r$. That restriction helps (less variance, fewer parameters) when the update the task needs is close to low-rank, and costs accuracy when it is not, whatever the norm of the change. On a real model this is an empirical question about the task and the layer, which is what §7 of the paper measures.

</details>
