# kazakh-parenting-llm

**A bilingual (Kazakh / Russian) assistant for parents of children aged 0–3:
QLoRA fine-tuning of an open-weights LLM + hybrid RAG + a reproducible evaluation of what each part adds.**

Parents in Kazakhstan ask about sleep, feeding, milestones and worrying symptoms in Kazakh, Russian or a mix of both.
Open models answer Russian reasonably well, but in Kazakh they often drift into Russian mid-answer, invent numbers,
or miss warning signs that need a doctor. This project builds the full pipeline to fix that on a single free GPU:

```
knowledge base (12 topics × kk/ru) ─► grounded synthetic Q&A ─► cleaning ─► QLoRA (Qwen3-4B, 4-bit) ─► GGUF
             │                                                                      │
             └──► hybrid retriever (BM25 + bge-m3, RRF, age-aware) ─────────────────┴─► FastAPI service
                                                                                         + safety guardrail
```

| | |
|---|---|
| Base model | `Qwen/Qwen3-4B-Instruct-2507` (switchable: Qwen2.5-7B, KazLLM-8B, any HF causal LM) |
| Fine-tuning | QLoRA: NF4 4-bit base, LoRA r=16 / α=32 on all attention + MLP projections, loss on answer tokens only |
| Retrieval | BM25 over prefix-stemmed tokens + `BAAI/bge-m3` dense, Reciprocal Rank Fusion, same-language filter, child-age boost |
| Personalisation | child profile in the system prompt: age, **corrected age for premature babies**, family notes (e.g. bilingual home) |
| Evaluation | 40 held-out hand-written questions (20 kk / 20 ru), 4 system variants, automatic metrics + LLM-as-judge |
| Serving | FastAPI · llama.cpp (Q4_K_M GGUF, CPU) or transformers + LoRA (GPU) · deterministic emergency guardrail |
| Hardware | everything runs on a free Kaggle T4 (16 GB) |

---

## Results

> Numbers below are produced by `notebooks/kaggle_pipeline.ipynb` and written to `results/`.
> Run the notebook and paste `results/summary.md` here — nothing in this section is filled in by hand.

| Variant | Key-fact coverage | Answer in requested language | Red-flag referral | Judge accuracy (1–5) |
|---|---|---|---|---|
| Base | _run notebook_ | | | |
| Base + RAG | | | | |
| QLoRA | | | | |
| QLoRA + RAG | | | | |

Retrieval ablation (already measured, BM25 only, on the **training** questions so the eval set stays untouched):

| Tokenisation | Recall@1 | Recall@4 |
|---|---|---|
| no stemming | 40 / 68 | 54 / 68 |
| 4-char prefix stemming (used) | **47 / 68** | **59 / 68** |

Kazakh is agglutinative (`бала → баланың, балаға, балалар`), so even crude prefix stemming gives BM25 a large boost;
the dense bge-m3 half of the hybrid covers the rest.

<!-- after the Kaggle run, uncomment:
![metrics](results/metrics.png)
![loss](results/train/loss.png)
-->

---

## What is evaluated, and why these metrics

Four variants answer the same held-out questions, so the contribution of RAG and of fine-tuning can be separated:
`base`, `base + RAG`, `QLoRA`, `QLoRA + RAG`.

| Metric | What it catches |
|---|---|
| **Key-fact coverage** | Each eval question has 1–4 hand-written key facts (stem lists, robust to inflection). Are the facts that matter actually in the answer? |
| **Answer-language share** | Share of sentences in the requested language. Detects the main Kazakh failure mode — switching to Russian halfway. Kazakh is detected with Kazakh-specific letters (ә ғ қ ң ө ұ ү һ і) + function words + suffixes, no extra model. |
| **Red-flag referral** | For questions with danger signs (fever under 3 months, swallowed battery, seizures, regression) the answer must send the parent to a doctor / 103 / 112. |
| **No dosing** | Out-of-scope medication questions must not get a dose. |
| **chrF** | Character n-gram F-score vs a gold answer — fairer than BLEU for Kazakh morphology. |
| **Retrieval recall@k** | Did the retriever surface a chunk from the right KB document? |
| **LLM-as-judge** (optional) | Accuracy, helpfulness, safety, language quality, 1–5, cached so reruns are free. |

The eval set is never used for training: `prepare_dataset.py` drops any training question with >0.6 character
5-gram Jaccard similarity to an eval question, and hyper-parameters (e.g. stem length) were tuned on training data.

## Data

* **Knowledge base** — 12 topics (sleep, feeding, motor & speech milestones, bilingualism, screens, crying/colic,
  safety, red flags, emotions/tantrums, play, toilet training, teeth), parallel in Russian and Kazakh, ~94 KB of text,
  124 chunks. Original summaries of WHO, UNICEF, CDC (2022 milestones), AAP, NHS and NICE guidance — see `data/kb/README.md`.
* **Seed set** — 72 hand-written Q&A (36 kk + 36 ru) with child profiles, including premature babies, bilingual families,
  emergencies and out-of-scope requests (medication doses, children older than 3).
* **Synthetic set** — the teacher model writes 4 questions per KB chunk per round in five parent "styles"
  (short messenger question, anxious and detailed, conflicting advice from relatives, typos...), answers grounded only
  in that chunk. Resumable, rate-limited for free API tiers.
* **Cleaning** — required fields, length bounds, question *and* answer language check, exact + near-duplicate removal
  (char 5-gram Jaccard), eval-leak guard. Drop statistics are saved to `data/processed/stats.json`.
* **Context mixing** — 50 % of training prompts include retrieved passages, 50 % do not, so the model learns both to use
  context and to answer without it.

## Design decisions

* **Completion-only loss, implemented by hand** (`src/kpl/sft.py`): prompt, profile and RAG passages are masked with -100.
  Otherwise the model spends capacity learning to reproduce retrieved text. Over-long examples are dropped, not truncated,
  so the model never learns to stop mid-sentence. Unit-tested with a stub tokenizer.
* **One prompt builder for train, eval and serving** (`src/kpl/prompts.py`) — no train/serve skew.
* **Same-language retrieval** — the KB is parallel, so a Kazakh question only gets Kazakh context; Russian context
  is the fastest way to make the model answer in Russian.
* **Age-aware ranking** — chunks whose age range covers the child get a small RRF boost.
* **Safety outside the model** — emergency patterns (not breathing, seizures, swallowed battery/magnet/pills...) are
  matched with regexes in both languages and an ambulance banner is prepended no matter what the model says
  (`src/kpl/guardrails.py`).
* **Corrected age** for babies born ≥3 weeks early, used until 24 months — a detail parents of premature babies
  care about and generic models usually ignore.
* **Merge into fp16, not into 4-bit** before GGUF export, to avoid compounding quantisation error.

## Quickstart

### Full pipeline on Kaggle (recommended)
Open `notebooks/kaggle_pipeline.ipynb` on Kaggle with **GPU T4 x2** and internet on, optionally add a
`TEACHER_API_KEY` secret (free Gemini key) and *Run all*. ≈1.5–2 hours.

### Locally, step by step
```bash
pip install -r requirements.txt
export TEACHER_API_KEY=...                      # optional: Gemini / Groq / OpenAI key, see configs/default.yaml

python scripts/build_index.py                   # chunks + BM25 + bge-m3      (--no-dense for CPU only)
python scripts/generate_synthetic.py --rounds 2 # grounded synthetic Q&A        (skip -> seed data only)
python scripts/prepare_dataset.py               # clean, mix context, split
python scripts/generate_answers.py --variants base base_rag   # baseline
python scripts/train_qlora.py                   # QLoRA, ~16 GB GPU
python scripts/generate_answers.py --variants ft ft_rag
python scripts/evaluate.py --judge              # tables + charts in results/
python scripts/merge_adapter.py && bash scripts/export_gguf.sh
```
Every config value can be overridden from the CLI, e.g.
`python scripts/train_qlora.py --set model.base=Qwen/Qwen2.5-7B-Instruct train.epochs=2`.

### Serve
```bash
llama-server -m artifacts/gguf/model-Q4_K_M.gguf --port 8080 -c 4096 &
uvicorn app.server:app --port 8000

curl -s localhost:8000/chat -H 'content-type: application/json' \
  -d '{"question": "Балам 10 айлық, түнде жиі оянады. Не істеймін?", "profile": {"age_months": 10}}'
```
Response: answer, detected language, emergency flag, cited KB documents, latency.
`KPL_BACKEND=hf` serves the 4-bit base + LoRA on GPU instead; `KPL_BACKEND=mock` runs without a model.

### Tests
```bash
pytest -q        # 36 tests, CPU only, < 1 s: language ID, chunking, retrieval, loss masking, data quality, metrics, API
```

## Repository layout
```
configs/default.yaml        single config for every step
data/kb/{ru,kk}/*.md        knowledge base (front matter: id, age range, sources)
data/seed/*.yaml            hand-written training examples
data/eval/eval.yaml         held-out evaluation set with key facts and gold answers
src/kpl/                    library: textutils, rag, prompts, sft, llm, metrics, guardrails, data, config
scripts/                    pipeline steps (index -> synthetic -> prepare -> train -> generate -> evaluate -> export)
app/server.py               FastAPI service
notebooks/                  Kaggle end-to-end notebook
tests/                      pytest suite
```

## Limitations and next steps
* Not medical advice. A production system needs every KB document reviewed by a pediatrician and answers monitored.
* 40 eval questions are enough to see large effects, not small ones; the next step is ~200 questions written together
  with a pediatrician, and human preference ratings from parents.
* The judge is itself an LLM; its Kazakh judgements should be spot-checked by native speakers.
* Next: DPO on judge-ranked answer pairs for tone and safety; video-lesson transcripts (Whisper) as an extra KB source;
  latency/throughput benchmark of GGUF Q4_K_M vs 4-bit transformers vs vLLM.

---

### Кратко по-русски
Ассистент для родителей детей 0–3 лет на казахском и русском. Полный цикл: своя база знаний на двух языках →
генерация и чистка датасета → QLoRA-дообучение Qwen3-4B на Kaggle T4 → гибридный RAG (BM25 + bge-m3) →
сравнение четырёх вариантов (база / база+RAG / QLoRA / QLoRA+RAG) по фактам, языку ответа и безопасности →
квантизация в GGUF и FastAPI-сервис с защитным слоем для экстренных ситуаций.

Author: **Askhat Amankulov** · MIT License
