# Knowledge base

12 topics × 2 languages (`ru/`, `kk/`), parallel by `id`. Each file has front matter:

```yaml
id: sleep                 # same id in both languages
title: ...
lang: ru | kk
age_months: 0-36          # used by the retriever to boost age-relevant chunks
sources: [...]            # guideline documents the text is based on
```

The texts are **original summaries written for this project** of public guidance from WHO, UNICEF,
CDC "Learn the Signs. Act Early." (2022 milestones), AAP / HealthyChildren.org, NHS and NICE.
No source text is copied. The Kazakh versions are translations of the Russian ones; terminology
follows everyday Kazakh used by parents (e.g. «еңбек» for fontanelle, «түбек» for potty).

To extend the KB, drop more `*.md` files with the same front matter into `ru/` or `kk/`
and re-run `scripts/build_index.py` (and `scripts/generate_synthetic.py` for more training data).

**Not medical advice.** This is a portfolio / research project. A production system for parents
would need review of every document by a pediatrician and continuous monitoring of answers.
