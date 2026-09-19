# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

A collection of standalone research notebooks studying how language models acquire, retain, and
lose factual knowledge, under two different mechanisms:

- **Parametric memory** — knowledge baked into model weights via LoRA fine-tuning
  ([fine_tuning_qwen_custom_data_lora.ipynb](fine_tuning_experiements/fine_tuning_qwen_custom_data_lora.ipynb),
  [fine_tune_qwen_gsm8k_lora.ipynb](fine_tuning_experiements/fine_tune_qwen_gsm8k_lora.ipynb))
- **Working/state memory** — knowledge held in an RNN's recurrent state (as opposed to a
  transformer's KV-cache/attention window), and how it decays under interfering context
  ([rnn_llm_analysis.ipynb](rnn_experiments/rnn_llm_analysis.ipynb))
- **Long-context "memory"** — whether a transformer that is simply handed its entire interaction
  history in-context (no fine-tuning, no retrieval) can still recall and reason over facts buried
  in it, reproducing an external benchmark's full experimental surface rather than one baseline
  finding
  ([long_mem_eval_v1_baseline.ipynb](paper_recreation/long_mem_eval_v1_baseline.ipynb))

There is no application code, package, build system, or test suite — the unit of work is a
Jupyter notebook. Most notebooks are written to run top-to-bottom in Google Colab (`!pip install`
their own deps in the first cells, assume a GPU runtime); `fine_tune_qwen_gsm8k_lora.ipynb`,
`fine_tuning_qwen_custom_data_lora.ipynb`, and `long_mem_eval_v1_baseline.ipynb` additionally
auto-detect Colab vs. local (`IN_COLAB = "google.colab" in sys.modules`) and run either way.
`rnn_llm_analysis.ipynb` is Colab-only by design (see its own subsection below).

## Environment

A local `.venv` (Python 3.12, managed by `uv`, with a CUDA-enabled `torch`) exists at the repo root
and is a legitimate place to actually run the Colab-portable notebooks listed above, not just edit
them — GPU work does not require Colab on this machine. Notebooks without Colab/local detection
still assume a Colab GPU runtime. When editing any notebook, check its own dependency-install cell
for the exact packages/versions it expects rather than assuming a shared environment across
notebooks — each notebook pins slightly different versions (e.g. `bitsandbytes>=0.46.1`,
`torchao>0.16.0`).

There is no lint/build/test tooling in this repo. "Running" a notebook (in Colab) *is* the
verification step for changes to it.

## Notebook-by-notebook architecture

### `fine_tuning_qwen_custom_data_lora.ipynb` — single-paragraph fact-learning experiment

The core experiment: fine-tune `Qwen2.5-1.5B-Instruct` on **one** short source paragraph of novel
facts, then test whether the model can answer differently-worded questions about those facts
(i.e. did it generalize the facts, or just memorize surface text). Designed to run once,
top-to-bottom, in a single pass. Runs in Colab or locally — an `IN_COLAB` check near the top
gates Drive mounting and picks paths accordingly (Drive under `IN_COLAB`, `./new_data` +
`./models/qwen-1_5b-learn-facts/` locally, matching `fine_tune_qwen_gsm8k_lora.ipynb`'s
`./models/...` convention) and only reinstalls `torch` locally if it's missing or lacks CUDA, so a
bare `pip install torch` doesn't clobber a working GPU build with PyPI's CPU-only wheel.

1. Load tokenizer + base model **once** — every later section (baseline scoring, LoRA training,
   post-training scoring) reuses these same in-memory objects. Do not reload the base model
   mid-notebook; that's what earlier buggy versions did and it invalidated the before/after
   comparison.
2. Score the **pristine** base model on a held-out QA set before any adapter exists (not via
   `disable_adapter()` on an already-wrapped model — that was the earlier bug).
3. Load the source paragraph, format it as a chat-turn training example with the prompt portion
   masked out of the loss (bare-paragraph training + chat-template eval was another earlier bug —
   it taught the model to continue prose instead of answer questions).
4. Attach a LoRA adapter (`r=16`, `alpha=32`, targeting attention *and* MLP
   `gate_proj`/`up_proj`/`down_proj` — MLP is included because that's "where facts live"), train,
   save adapter weights to disk.
5. **Reload the adapter from disk** (not the in-memory trained object) before final scoring, so
   the reported number reflects the actual saved artifact.
6. Compare `base_score` vs `tuned_score` on the same held-out QA set.

Key data-format contract this notebook depends on (see `new_data/`):
- A "paragraph" file (`*_training.json` or `*_paragraph_training.json`): `{"text": "..."}`, a
  single dense paragraph of facts about one subject.
- A matching "QA" file (`*_qa.jsonl` or `*_qa_test.json`): one JSON object per line,
  `{"prompt": ..., "completion": ..., "aliases": [...]}` — `aliases` lists acceptable alternate
  phrasings of the correct answer, used by the scoring harness for fuzzy/alias matching instead
  of exact string match.

`new_data/` holds the current generation of paragraph+QA pairs (Shajee/Game-of-Thrones persona,
Forza Horizon 6, Fears to Fathom). `old_data/` holds an earlier data format for the same idea
(`shajee_facts_corpus.json` + `shajee_got_train_prompt_completion.json`, many prompt/completion
pairs instead of one paragraph) — treat `old_data/` as superseded reference, not the current
pipeline input.

### `fine_tune_qwen_gsm8k_lora.ipynb` — LoRA fine-tuning on GSM8K math, base-vs-tuned comparison

Answers a specific question: does LoRA fine-tuning `Qwen2.5-1.5B-Instruct` on GSM8K's official
train split actually move accuracy on GSM8K's official (held-out) test split, and by how much can
that really be claimed given sampling noise? Structured like the custom-data notebook — load the
tokenizer/base model once, score the pristine base model before any adapter exists, train, reload
the adapter **from disk** (not the in-memory trained object), score again, compare — but built
from scratch rather than adapted from an earlier version, after auditing an older draft that
trained on a hand-rolled slice of *train* (never touching the real *test* split), had no
evaluation code at all beyond two unrun qualitative prints, and used `</s>` as an end-of-sequence
marker (which is not a special token for Qwen — it tokenizes as three ordinary text tokens).

Key design points worth knowing before editing this notebook:
- **Fairness of the comparison**: base and tuned are both prompted through the same function
  (`to_chat_example`), Qwen's own chat template plus one pinned `SYSTEM_PROMPT` instructing
  `#### <number>`-terminated answers. This function is reused to build training examples too, so
  the training prompt and the eval prompt are identical by construction — there's no separate
  "training format" that could drift from the "eval format." Qwen's template silently injects its
  own default system message when none is given, which is exactly the kind of drift this avoids.
- **Honesty about what's being measured**: GSM8K is a heavily-used SFT dataset and
  `Qwen2.5-1.5B-Instruct` was already instruction-tuned before this notebook touches it, so this
  is closer to a format/style-adaptation experiment than a "teach it new math" experiment — and
  GSM8K's gold rationales are much terser than Qwen's native chain-of-thought, so a flat or
  negative accuracy result is an expected, legitimate possible outcome, not a bug. The notebook
  says this up front rather than only in a caveats section.
- **Statistics are paired**, not independent-sample: both models are scored on the same questions
  in the same order, so the comparison uses Wilson score intervals per model plus an exact
  (scipy-free, `math.comb`-based) McNemar test on the discordant pairs — overlapping marginal
  confidence intervals do not imply "no difference" in a paired design.
- **No `bitsandbytes`/4-bit and no `trl`**: the model fits comfortably in bf16 on a 16GB GPU, so
  4-bit quantization (which costs speed and blocks a clean `merge_and_unload()`) is skipped by
  default (`USE_4BIT` flag kept for scaling to larger models). Training examples are built and
  loss-masked by hand (chat-template prompt + `-100` over the prompt span, matching the sibling
  notebook's pattern) and trained with a plain `Trainer`, rather than `SFTTrainer`/TRL — keeping
  the exact string a model is trained on fully visible rather than hidden behind a training
  framework's internal templating.
- **Portable**: GPU/dtype (`bf16` vs `fp16`) and Colab-vs-local output paths are auto-detected;
  there's a `SMOKE` flag in the config cell for a fast end-to-end plumbing check before committing
  to a full run.

### `rnn_llm_analysis.ipynb` — how long a fact survives in RWKV's recurrent state

**Colab-only** (it hard-asserts `"google.colab" in sys.modules` — there is deliberately no
local/Windows code path to keep in sync). Uses **RWKV-7 World 1.5B** via the inference-only `rwkv`
pip package and the `.pth` checkpoint, *not* the HF/`fla` path used by the fine-tuning notebooks:
this is pure inference with explicit state passing, and the pip package runs it in `fp16` on a free
Colab T4, whereas `fla` would force bf16 + Triton and an L4/A100.

Every probe has the same shape — `FACT → NOISE → QUESTION` — and the sweep runs noise from **0 to
25,000 words** (~34k tokens, ~8× the checkpoint's 4,096 training context). 4 conditions × 20 fact
items × 13 budgets = **1,040 probes**, ~25-30 min on a T4.

Design points worth knowing before editing:
- `RWKV_V7_ON=1` **must** be set as an env var before `from rwkv.model import RWKV` is imported —
  the package only switches to the v7 (`RWKV_x070`) implementation when this is set at import
  time; a v7 checkpoint loaded under the default v4/5/6 code path silently misbehaves rather than
  erroring.
- **`RWKV.forward()` does not chunk internally** (`seq_mode = len(tokens) > 1`, then the whole list
  goes through the sequence kernels at once), so `feed()` splits into `CHUNK_TOKENS=1024` pieces and
  threads the state through. This is a memory guard, not an approximation.
- **The sweep is single-pass.** Budget *k* is a strict word-prefix of budget *k+1*, so each
  condition-item feeds its 25,000-word noise stream **once**, probing from a cloned state at each
  checkpoint — 25,000 words instead of the naive 79,350, exactly equivalent. Both this and the
  chunking rest on state-passing associativity, which section 5 tests explicitly (comparing
  **argmax**, not floats — float addition is not associative). Do not weaken that cell.
- `answer(state, ...)` greedy-decodes from a **clone**, so probing recall at one budget never
  contaminates the state carried into the next.
- Four conditions: `control` (fact withheld, wikitext noise — the chance floor, and a *matched*
  counterfactual reusing the same noise window as `natural`), `repeat` (near-zero-entropy filler),
  `natural` (WikiText-2, de-Moses-tokenised so it reads as real prose), `interference`
  (**adjacent-slot** claims — "My surname is …", "My dog's name is …" — which state *a* name but
  not *my* name). Scored `correct` / `distractor` / `other`; the `distractor` bucket is what
  separates "forgot it" from "confidently answered with a name out of the noise".
- Target and distractor name pools are asserted disjoint, and matching is whole-word after
  lowercasing, so a chance substring cannot score.
- Statistics are **Wilson score intervals** (n=20 per cell, and the run spends most of its time at
  0/20 and 20/20 where the normal approximation gives a zero-width interval). Reported alongside the
  budget at which each curve crosses below 90% / 50%.
- Figure 1 plots recall against **both** words and tokens: the arms have different tokens-per-word
  ratios, so equal words is not equal work through the state, and the token panel is the fair
  cross-arm comparison. The x-axis is `symlog` (so the 0-noise point plots) with pinned limits —
  symlog is symmetric about zero and will otherwise draw empty negative decades.

Deliberately **not** in this notebook, having been cut from an earlier version: the per-layer
cosine-distance trace and its heatmap (it needs a second parallel noise-only pass per item, roughly
doubling runtime for a mechanistic side-question), and the half-finished Mamba comparison. The
pre-rewrite version is kept at `rnn_llm_analysis.pre-rewrite-2026-08-11.ipynb`.

### `paper_recreation/long_mem_eval_v1_baseline.ipynb` — full LongMemEval reproduction

Reproduces the **whole** experimentally-reproducible surface of *LongMemEval: Benchmarking Chat
Assistants on Long-Term Interactive Memory* (Wu et al., ICLR 2025, arXiv:2410.10813v2) in one
notebook — not just §3.4's long-context reading baseline (the predecessor version's scope), but
also §5's memory-design experiments: value granularity (§5.2), key expansion (§5.3, Table 3),
time-aware query expansion (§5.4, Table 4), reading strategy (§5.5, Figure 6), and a combined
best-recipe run (§5.6). Out of scope: §2's benchmark construction (consumes the released dataset
instead) and §3.3's commercial-assistant study (ChatGPT/Coze — needs live manual UI interaction).

**Reader coverage is capped by what fits one Colab A100-40GB pulled from HuggingFace**: only 3 of
the paper's 5 Figure-3b reader rows (`Llama-3.1-8B-Instruct` bf16, `Phi-3.5-mini-instruct` and
`Phi-3-medium-128k-instruct` both needing `kv_cache_dtype="fp8"`) and 1 of its 3 memory-design
reader columns (`Llama-3.1-8B-Instruct`) — GPT-4o isn't on HuggingFace and Llama-3.1-70B needs
~140GB. GPT-4o is retained only as the **judge** (`gpt-4o-2024-08-06`, matching the paper's own
evaluator) and as the strong extractor in the time-aware query expansion ablation. Every *relative*
finding the paper reports (oracle→S drop, key expansion's recall gain, time-aware expansion's
temporal gain including the weak-extractor's negative result, CoN+JSON's reading gain) is
reproducible on the Llama-3.1-8B column; absolute GPT-4o/70B rows are reported from the paper for
context only, never claimed as reproduced.

**Runs in Colab (primary) or locally (secondary, whatever fits the GPU)** — `IN_COLAB =
"google.colab" in sys.modules` gates Drive mounting and the work-directory root, matching the
`IN_COLAB` convention shared with `fine_tune_qwen_gsm8k_lora.ipynb` and
`fine_tuning_qwen_custom_data_lora.ipynb`. Capability gates (`CAN_RUN_LONGCONTEXT_8B`,
`CAN_RUN_M_RETRIEVAL`, etc.) derived from detected VRAM let an experiment family that doesn't fit
the current GPU skip itself with a printed reason instead of crashing, so a 16GB local card still
runs the retrieval, oracle-reading, and RAG-QA families end to end even though the long-context S
arms need an A100. **Fully self-contained**: no clone of the authors' repo
(`github.com/xiaowu0162/LongMemEval`), no `.env`, no sibling files — every prompt template, the JSON
history-serialization format, the truncation rule, all five question-type-specific LLM-judge
prompts, the retrieval/NDCG math, and the key-expansion/time-range-extraction prompts are
transcribed inline, because the authors' own scripts assume a Slurm-style multi-GPU cluster and pin
package versions (`vllm==0.5.3.post1`, `transformers==4.43.3`) that don't install on current Colab.

Fidelity is protected without vendoring code: a Gate 2 cell fetches the authors' current source live
from `raw.githubusercontent.com` via plain `urllib` (into memory, no clone) across all seven source
files this notebook draws from, and asserts every transcribed string/rule appears verbatim —
warning rather than failing if GitHub is unreachable, so a flaky network never blocks a run.

Two structural decisions make the M-scale (`longmemeval_m_cleaned.json`, 2.74GB, ~500
sessions/question) experiments tractable on a laptop instead of a cluster:
- **Datasets are normalized on first load, not held in RAM.** Every released file is streamed once
  via `ijson` into `store/sessions.jsonl` (one row per *unique* session body, deduped by
  `sha1` of its canonical JSON) and `store/instances_<dataset>.jsonl` (per-question metadata only,
  referencing sessions by hash). This is the difference between needing 60GB of RAM and running on
  a laptop.
- **Embeddings are cached by text hash, not by (question, session).** The authors' own
  `run_retrieval.py` re-embeds a question's entire haystack from scratch per question (correct on
  their multi-GPU cluster, wasteful on one GPU); since embeddings are a pure function of text, this
  notebook embeds each unique string once (`EmbeddingCache`, keyed by `sha1(text)`) and looks it up
  thereafter. Both the embedding-cache round-trip and the deduped path's equivalence to the authors'
  literal per-question path are **asserted in Gate 3**, not assumed.

Structured like the other notebooks in this repo: numbered sections (1–20), a Gate 0–5 sequence
(preflight → dataset integrity → prompt fidelity → retrieval-harness self-check → smoke, implicit in
`PRESET="smoke"` rather than a parallel code path → full-500 oracle sanity, warn not fail) before
committing GPU time to any experiment, resumable chunked JSONL writers keyed by `question_id` under
`runs/<experiment>/` so a dropped Colab session only loses the in-flight chunk, a `ModelManager`
that holds at most one vLLM engine at a time (tearing the previous one down before loading the
next — 40GB has no room for two), and a reporting section with paired McNemar tests plus Wilson
score intervals per question type — same statistical conventions as `fine_tune_qwen_gsm8k_lora.ipynb`.
The primary pass/fail for the long-context arm is still the oracle→S *relative* accuracy drop, not
the absolute accuracy numbers, since Figure 3b and Appendix Table 8 disagree with each other on
absolutes and are reported against both rather than reconciled.

`PRESET` (`"smoke"` / `"core"` / `"full"`, in the config cell) controls scale end to end: `"core"`
(the default) is the full 500-question benchmark with `{Llama-3.1-8B-Instruct, Phi-3.5-mini-instruct}`
as readers and Stella V5 as the retriever (the paper's own §5.1 choice); `"full"` adds
`Phi-3-medium-128k-instruct`, the BM25/Contriever/GTE retriever sweep, the `nl` history format, and
`con-separate` reading. §20 has the full runtime/cost breakdown (~30-40 A100-hours, ~$15-18 in
OpenAI judge/extraction calls under `"core"`) and the complete list of known deviations from the
authors' exact methodology (fp8 KV cache for two of the three readers, abbreviated ICL example sets
in the key-expansion regeneration fallback, undocumented internal key format in the authors'
released expansion-cache archives).

### `longmemeval_stream_state.ipynb` — incremental vs. re-ingested secondary-model memory

Extends the RWKV-first, Llama-second pairing built in `longmemeval_memorag_ext_new.ipynb` (RWKV-7
G1 sees the query first and hands back clues/a draft answer; a Llama model reads them — either
directly, `memonly`, or as retrieval queries over the session corpus, `memorag_top{5,10}` — and
writes the final answer). That notebook rebuilds RWKV's state from the **entire** history for
every query; this one asks whether the state instead has to be maintained **incrementally** —
folded forward one session at a time and persisted between queries, the way a deployed assistant
actually receives conversation — and whether that changes the clues handed back, and therefore
what the primary model answers. The pairing itself never changes: every arm is RWKV-first,
Llama-second, and there is no RWKV-alone arm. What varies is the **policy** used to build the
secondary model's state (`stream` — one persisted state, probed from a clone; `bank` — several
snapshots retained from that one pass, clues unioned; `segment` — state reset periodically to stay
inside RWKV's trained context; `reingest` — ext_new's own from-scratch rebuild, kept only as a
correctness **gate**, never run as a full accuracy arm) crossed with the **route** the clues take
to the generator (`memonly` vs. `memorag_top5`, primary — ext_new measured top5 beating top10 at
every RWKV size for both readers).

`stream ≡ reingest` is a theorem in exact arithmetic (RWKV-7's update is a deterministic fold), so
it is treated as a gate, not a finding — a tight match with small McNemar b/c counts is this
notebook's *success* condition. What is not a theorem — fp16 drift over ~50 session boundaries,
greedy-decode divergence from one flipped logit, a live state-aliasing bug class (`model.forward`
mutates the state list you pass it, so a retained snapshot that skips `[t.clone() for t in state]`
silently degenerates `bank` into `stream` without erroring), and disk round-trip fidelity — is
measured by Gates G6–G10, whose headline output, `NOTE_DIVERGENCE_RATE`, is cited as the noise
floor in every later figure caption. `memonly` has almost no headroom on its own (ext_new measured
it at 0.14–0.15 against a 0.08 closedbook floor), so the decay analysis leads with a judge-free
evidence-retrieval-recall metric rather than end-to-end accuracy, and its headline statistic is a
within-question paired drop (each question's own Δ=0 vs. its own largest reachable Δ), not the
pooled decay curve — the naively pooled curve is measurably biased toward the easy single-session
questions that survive to large Δ (measured: 51% of `single-session-user` questions reach Δ=20
sessions past their evidence, vs. 18–20% of `multi-session`/`temporal-reasoning`/
`knowledge-update`).

Two tiers: **Stage 1** reuses ext_new's own 100-question subset (Gate 4 asserts byte-identity with
its cached `question_subset_100.json`, which is what licenses every McNemar comparison against its
published numbers) as 100 independent native streams. **Stage 2** merges groups of 8 of those
questions (pairwise-disjoint evidence, no shared `answer_session_id`, `_abs` questions excluded)
into much longer chronological mega-streams (~380 sessions, ~900k tokens — 45–60× RWKV's trained
context) so that large-Δ decay and question-level amortization become measurable at all; merging
is a strictly harder retrieval problem for reasons unrelated to state persistence (measured: ~70%
of merged questions acquire a co-merged session containing their own gold answer's key terms), so
**Stage-2 accuracy is never compared to Stage-1, to ext_new, or to any published LongMemEval
number** — only within Stage 2, and the per-question distractor density is logged as a covariate
rather than hidden. Baselines (`closedbook`/`oracle_con`/`s_con`/`rag_q_top5`) are **imported**
from ext_new's own `judged.jsonl`, never re-run, since Gate 4's subset guarantee makes that valid
and it costs zero extra API spend.

`longmemeval_memorag_ext_new.ipynb` is read-only from here (its `ext/` results, its shared
`store/` — dataset, tokenizers, question subset, sha1-keyed Stella embedding cache); this
notebook's own outputs live under a parallel `stream/` tree so the two can never collide. Every
transcribed primitive (env/deps/dataset/prompt-fidelity gates, `feed`/`probe_memory`, the
retrieval stack, generation backends, judging, Wilson/McNemar) is copied in verbatim rather than
imported, per this repo's standalone-notebook convention — a name-resolution sweep and a
cell-by-cell diff against ext_new are how that transcription is checked before trusting a run.

## Working with the data files

When adding a new subject for the fact-learning experiment, follow the existing pair pattern in
`new_data/`: write one dense source paragraph as `{"text": ...}`, then write a held-out QA set as
JSONL with `prompt`/`completion`/`aliases` per line, covering the facts in the paragraph but
phrased differently than the paragraph's own wording (the whole point of the experiment is
generalization, not substring recall).
