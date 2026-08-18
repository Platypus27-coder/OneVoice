# OneVoice Construction Dataset v2

## What changed from v1

This version is designed as a **controlled synthetic bootstrap dataset**, not merely a larger phrase list.

### Data layers

- `terminology_master.csv`: 216 construction concepts across 12 domains.
- `term_aliases.csv`: 426 Vietnamese standard/alias/slang surface forms.
- `utterances_core.csv`: 864 conservative core VI↔EN utterances.
- `utterances_augmented.csv`: 7200 controlled variations: spoken, terse, code-switch, warning, numeric and measurement language.
- `utterances_all.csv`: combined utterance corpus.
- `train.csv`, `dev.csv`, `test.csv`: split by **frame pattern**, not by random sentence.
- `safety_fast_path.csv`: 196 safety-message candidates.
- `minimal_pairs.csv`: 240 critical contrasts for negation, number and direction preservation.
- `asr_confusion_candidates.csv`: 237 ASR normalization/confusion candidates. These are NOT observed errors.
- `site_pack_template.json`: site-specific extension schema.
- `stats.json`: dataset statistics and leakage checks.

## Important label semantics

- `synthetic=true`: generated/curated by AI; not field-observed.
- `field_observed=false`: never claim this row was actually spoken on a worksite.
- `review_status=needs_domain_expert_review`: requires engineer/site review.
- Safety rows require a safety officer before production use.
- ASR confusion candidates require actual model measurements before being labeled as real errors.

## Split design

The split unit is `frame_pattern_id`. All examples generated from the same pattern stay in only one split.
This reduces exact template leakage between train/dev/test.

Current integrity:
- Frame-pattern leakage: 0
- Exact Vietnamese sentence leakage across splits: 0

## Recommended use

1. Review `terminology_master.csv` and aliases with a construction engineer.
2. Freeze `utterances_core.csv` after review; use it as a candidate evaluation corpus.
3. Use `train.csv` for domain adaptation / classifiers / terminology-aware MT.
4. Do **not** treat synthetic `dev.csv` / `test.csv` as proof of real-site performance.
5. Build a separate real-site benchmark later.
6. Use `minimal_pairs.csv` to test safety-critical semantic preservation.
7. Use `safety_fast_path.csv` only after safety review.
8. Use `asr_confusion_candidates.csv` for targeted testing, then replace candidates with errors actually observed from your ASR.

## Source basis

The dataset uses official sources to define the domain/hazard vocabulary and scope:
- Vietnam QCVN 18:2021/BXD: https://vbpl.vn/TW/Pages/vbpq-toanvan.aspx?ItemID=152388
- OSHA Construction Glossary: https://www.osha.gov/etools/construction/glossary
- OSHA Focus Four: https://www.osha.gov/training/outreach/construction/focus-four
- OSHA Trenching & Excavation: https://www.osha.gov/etools/construction/trenching/
- OSHA Scaffolding: https://www.osha.gov/scaffolding/construction
- OSHA Welding/Cutting: https://www.osha.gov/laws-regs/regulations/standardnumber/1926/1926.350
- OSHA Concrete/Masonry definitions: https://www.osha.gov/laws-regs/regulations/standardnumber/1926/1926.700

**Vietnamese↔English translations and site slang mappings are synthetic bootstrap content, not official translations from those sources.**

## Do not do this

- Do not train a production ASR from scratch using this text-only corpus.
- Do not report synthetic-test accuracy as real-world construction-site accuracy.
- Do not mark hypothetical ASR confusions as observed errors.
- Do not ship fixed safety translations without a competent safety/domain review.
