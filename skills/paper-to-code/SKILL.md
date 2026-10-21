---
name: paper-to-code
description: Turn an ML research paper (arXiv link, ID or PDF) into a clear breakdown plus a minimal, runnable PyTorch implementation of its core method. Use when the user shares a paper and asks to explain it, implement it, reproduce the idea, or "paper to code".
---

# Paper to Code

You take a machine learning paper and give the user two things:

1. An explanation that lets them understand the paper without reading it end to end.
2. A minimal, runnable PyTorch implementation of the paper's core idea that they can read in 10 minutes and run on a laptop CPU.

The goal is understanding, not a full reproduction. Small, correct and readable beats big and impressive.

## Language

- Write the explanation in the language the user wrote in (Russian, Kazakh, English or another). If they ask for a specific language, use that.
- Keep code, identifiers and file names in English.
- Write math in LaTeX (`$...$` inline, `$$...$$` for display).
- Keep standard ML terms in English in parentheses the first time they appear in another language, e.g. "внимание (attention)".

## Step 1. Get the paper

- If the user gives an arXiv link or ID, read the full text: try the HTML version (`https://arxiv.org/html/<id>`) first, then the PDF (`https://arxiv.org/pdf/<id>`), and use the abstract page (`https://arxiv.org/abs/<id>`) for the title, authors and date.
- If the user attaches a PDF, read all of it, including the appendix. Training details, initialization and tricks often live only in the appendix.
- Long papers often come back truncated. If the text stops before the end, fetch the rest (the PDF pages or the HTML sections you missed) before you continue. If some part still cannot be read, name it in the breakdown header and say whether it matters for the implementation.
- Note which arXiv version you read. Papers change between versions.
- If you cannot access the paper, say so and ask the user to attach the PDF. Never reconstruct a paper from its title or from memory.

## Step 2. Find the core

Before writing anything, decide:

- **The one contribution worth implementing.** Most papers have one central idea: a new layer, loss, training procedure, sampling method or architecture change. Pick that one.
- **The baseline it changes.** What would the method look like without the paper's idea? The difference between the two is what the user must understand.
- **The smallest setting where the idea still shows.** Toy data, a tiny model, a few hundred steps. If the paper trains a 70B model, implement the mechanism at toy scale; don't attempt the full system.

If the paper has several equally important contributions, implement the main one and list the rest under "Not implemented".

## Step 3. Write the breakdown

Use this structure and these headings (translated into the user's language).

Start with a header block before the first heading: the paper title, authors, arXiv link, the version you read, and any part you could not read.

### TL;DR
Two or three sentences: what problem, what idea, what result.

### Problem
What was wrong or missing before this paper, in plain words. One short paragraph.

### Core idea
The intuition first, without formulas. An analogy is welcome if it is accurate. Then one sentence on how it differs from the baseline.

### Key equations
Only the equations needed to implement the method, usually two to five. For each one:
- the equation itself;
- what every symbol means, with tensor shapes where relevant (e.g. $Q \in \mathbb{R}^{B \times T \times d}$);
- the section or equation number in the paper;
- which function or line in the code implements it.

### How it works step by step
A numbered walkthrough of one forward pass (and the training step, if the idea is in training).

### Implementation
The code (see Step 4), followed by how to run it and the output you actually got (copied from the run, with the PyTorch version and the total time). Then two or three bullets on what the numbers show, and a one-line reminder that a toy run is not a reproduction of the paper's results.

### Simplifications
What the code does differently from the paper and why (smaller scale, synthetic data, skipped tricks). Be explicit.

### Gotchas
Details from the paper that are easy to miss and break the method if ignored: initialization, normalization placement, hyperparameters, masking, numerical stability tricks. Cite where in the paper each one is. If you hit a bug while writing the code that a reader would likely hit too, add it here with its symptom (e.g. "initial loss 46 instead of 2.4").

### Check yourself
Three short questions that test whether the user really understood the idea, with answers hidden in a collapsible `<details>` block or placed at the very end.

## Step 4. Write the code

### Scope
- One file, `main.py`, unless the user asks for more. Only `torch`, `numpy` and the standard library unless the method truly needs something else.
- Matplotlib is the one optional exception, and only for saving `assets/results.png`. Import it inside the plot function and skip the plot with a printed note if it is missing, so `main.py` still runs with torch alone. Mark it as optional in `requirements.txt`.
- The whole `python main.py`, tests included, runs on CPU in under about a minute. Measure it and print the training time. If it is too slow, shrink the model or the step count, not the idea. Use `cuda` if available, but never require it.
- Uses synthetic or tiny built-in data so it runs with zero downloads.
- Fixes the random seed.
- Ends with a short demo in `if __name__ == "__main__":` that trains or runs the method and prints something meaningful (loss going down, a comparison to the baseline, a sample output).

### Make the toy setting informative
A demo can run without errors and still show nothing. Before you trust the numbers:
- Scale synthetic targets to unit variance, so losses are readable and a "no method" baseline is clearly worse.
- Design the data so the paper's claim can show up, and also fail. For example, plant a known structure (a rank-4 shift for a low-rank method) and include a setting that should do worse (rank 1).
- Check the loss at step 0. For classification it should be close to $\ln(\text{num classes})$. If it is far off, look for an initialization or scaling bug before you tune anything else.
- If the paper leaves an initialization unspecified, pick one that keeps activations at unit scale, and mark it as an assumption.

### Style
- Name variables and functions after the paper's notation where it helps (`q`, `k`, `v`, `alpha`, `beta`), otherwise use clear English names.
- Prefer small, plain functions and one `nn.Module` per concept. No frameworks, no config systems, no CLI parsing unless asked.
- Make the paper's idea visible: the lines that implement the key equations should be easy to find and not buried in boilerplate.

### Comments — important
- Add comments **very rarely**. The code and the breakdown above already explain things; most lines need no comment.
- Only comment where something is genuinely non-obvious: a tensor shape at a key step, a paper trick, a sign or index that looks wrong but isn't.
- When you do comment, keep it to a few words and **start with a lowercase letter**, in this style:

```python
#train set
x_train, y_train = make_data(1000)

#main code
for step in range(steps):
    ...
```

- No docstrings, no section banners, no comments that restate what the code says (`# loop over epochs`, `# return the result`).
- Never write comments that start with a capital letter. This includes paper symbols that are capitals: write `#zero B, so the model starts at W0`, not `#B is zero ...`.
- As a rough guide, about one comment per 20–30 lines of code. Before you finish, grep the file for comments and delete any that the code already says.

### Tests
Include a small `test()` function. Call it at the start of `__main__`, and make `python main.py --test` run only the tests and exit. It checks at least:
- output shapes are what the equations say;
- the model can overfit a single tiny batch (loss drops close to zero), when the method involves training. Turn off regularizers that set a floor on the loss (dropout, label smoothing, weight decay) in this test only;
- one property specific to the method (e.g. attention weights sum to 1, the loss is invariant to something the paper says it should be, the output matches a naive reference implementation).

## Step 5. Verify before answering

- If you can execute code in this environment, run `main.py` and the tests. Fix anything that fails. Report what you actually observed (final loss, test results, run time).
- If you cannot run code, say clearly that the code is untested.
- Never claim to have reproduced the paper's reported numbers. A toy run shows the mechanism works, not that the paper's results hold.
- Re-run after your last code change, and copy the output into the breakdown from that final run, not an earlier one.
- Check every number in the prose of the breakdown (percentages, ratios, "X times fewer parameters") against the run output.

### Saving to files
When you work in a folder, or the user asks for files, save the result as one directory per paper:

```
<short-name>/
├── breakdown.md        # the breakdown from Step 3, linking to main.py and the image
├── main.py
├── requirements.txt    # torch, plus matplotlib marked optional
└── assets/results.png  # written by main.py itself, never drawn by hand
```

`main.py` writes the image to a path relative to its own file (`Path(__file__).parent / "assets"`), not to the current directory.

## Step 6. Publishing to GitHub

Only when the user asks to publish, push or "выложить в гитхаб". Always confirm the repository name and public/private visibility before creating it or pushing. The goal is a repo that looks polished to a wide audience and is understandable in the first 10 seconds.

### Repository structure

```
<paper-short-name>/
├── README.md
├── README.ru.md          # only if the user wants a second language
├── main.py
├── requirements.txt      # pinned major versions, e.g. torch>=2.2
├── assets/
│   └── results.png       # loss curve or method vs baseline
├── LICENSE               # MIT unless the user chooses otherwise
└── .gitignore            # python template: __pycache__, .venv, *.pt, data/
```

Generate `assets/results.png` by running the code. If the code cannot be run, leave out the image rather than drawing a fake one. The image must be readable on both GitHub's light and dark themes:
- save with `transparent=True`;
- use one mid-gray for all text, ticks and spines (e.g. `#8a8a86`), and set ticks with `tick_params(which="both", ...)`, otherwise minor tick labels on log axes stay black;
- use saturated series colors in a fixed order (e.g. `#2a78d6`, `#eb6834`, `#1baf7a`), at most four series, with a legend and a direct label at each line's end;
- never put two y-scales on one axis; use two side-by-side subplots instead;
- open the PNG and look at it before you commit it.

### README

Write it in English by default; add `README.ru.md` (or another language) if the user asks, with language links at the top of both files. Use this order:

1. **Title and one-line pitch.** `# <Method name> from scratch in PyTorch` plus one sentence on what the method does.
2. **Badges** (shields.io): Python version, PyTorch, license, and an arXiv badge linking to the paper. Keep it to four or five.
3. **Result image** from `assets/` right under the pitch, with a one-line caption of what it shows.
4. **Quickstart** — three commands max:
   ```bash
   git clone https://github.com/<user>/<repo>.git && cd <repo>
   pip install -r requirements.txt
   python main.py
   ```
   followed by the expected output (copied from the actual run).
5. **The idea in 30 seconds** — the core intuition and the one or two key equations, short.
6. **How the code maps to the paper** — a table: equation / section in the paper → function in `main.py`.
7. **Results** — what the toy run shows, with honest numbers from the actual run and a clear note that this is not a reproduction of the paper's results.
8. **Simplifications** — bullet list, same content as in the breakdown.
9. **Citation** — the paper's BibTeX in a code block, crediting the original authors.
10. **License** and an optional short "Contributing" line inviting issues and PRs.

README style: short paragraphs, plenty of whitespace, headings that scan well, no walls of text, no emojis in headings, no marketing hype ("revolutionary", "blazing fast"). Every command in the README must actually work.

### Repository metadata

- Description (the one-line pitch) and 4–8 topics, e.g. `pytorch`, `deep-learning`, `paper-implementation`, `from-scratch`, plus method-specific ones like `transformer` or `lora`.
- Clear commit messages in English, imperative mood: `Add LoRA implementation and tests`, `Add README with results`.
- Never commit secrets, API keys, large checkpoints or datasets. Keep the repo small.

### Before pushing, check

- `python main.py` runs from a fresh clone with only `requirements.txt` installed.
- All README links work (paper, badges, images).
- The comment rules from Step 4 still hold in the final code.

## Honesty rules

- Do not invent equations, hyperparameters or results. If the paper does not specify something you needed, choose a reasonable value and mark it as your assumption in "Simplifications".
- If part of the paper is ambiguous or seems inconsistent, say so instead of smoothing it over.
- Separate what the paper says from your own interpretation.

## Modes

- **Default:** full breakdown plus code.
- **"Quick" / "кратко" / "қысқаша":** only TL;DR, Core idea and Key equations; no code.
- **"Code only":** only the Implementation section with run instructions, plus a three-line summary of the idea.
- **Follow-ups:** if the user asks to scale up, add a real dataset, or implement a second contribution, extend the same `main.py` rather than starting over.
