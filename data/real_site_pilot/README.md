# Real-site pilot contract

Target: 1,000 utterances (allowed range 500–2,000). Synthetic and real-site reports must remain separate.

Each recording requires the fields defined by `schema.json`. Split assignment must be grouped by
`site_id + session_id + speaker_id`; a speaker/session/site group may occur in only one split. Rows marked `test`
form the fixed final holdout and must never be used for model training, prompt selection or threshold tuning.

Audio collection requires project approval and participant consent. Do not store unnecessary personal identifiers in
the manifest; use pseudonymous speaker IDs.

Audit and lock the fixed test holdout before any tuning:

```bash
python scripts/audit_real_site.py --manifest /path/manifest.jsonl \
  --report reports/real_site/audit.json --create-holdout-lock /path/holdout_lock.json
```

Subsequent runs must pass `--holdout-lock /path/holdout_lock.json`. Synthetic reports must not be merged into this
report.
