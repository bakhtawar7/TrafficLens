# Changelog

## 2026-11-13

- Files: `core/evaluation.py`, `tests/test_evaluation.py`, `tests/test_adjudication.py`, `scripts/make_adjudication_list.py`, `docs/CHANGELOG.md`.
- Reason: preserve original and corrected label values for rows marked `adjudicated=yes`, compare evaluator results with and without adjudicated rows, test two-pass category/covered-total agreement with scene-level bootstrap intervals, and provide a readiness-gated id-only adjudication-list generator.
- Adjudicated label rows keep corrected counts in the regular category columns and preserve the prior values in matching `original_<category>` columns.
