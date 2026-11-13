# Test report and verification status

## Execution status

**No test suite was executed during this documentation-only work.** This is intentional:
the project owner instructed that tests must not be run for now. Therefore this report does
not claim that the current checkout passes, and it does not provide fabricated suite counts,
durations, or pass/fail totals.

A test-passing constraints file has not been created. `requirements.txt` currently contains
minimum dependency versions; exact versions that passed the suite cannot be asserted until
the owner authorizes a clean-environment run.

## Test command

From the repository root, after installing `requirements.txt` in an isolated virtual
environment:

```bash
python -m pytest
```

For an individual test module:

```bash
python -m pytest tests/test_history.py
python -m pytest tests/test_image_cases.py
```

These are instructions only; neither command was executed for this report.

## Test inventory

The repository's `tests/` directory contains the following areas. The table is an inventory
based on filenames and source inspection, not an assertion that every test currently passes.

| Test module | Apparent coverage from module name / inspection |
| --- | --- |
| `test_adjudication.py` | Human label adjudication |
| `test_annotate.py` | Detection annotation rendering |
| `test_app.py` | Streamlit app-level behaviour and limitations text |
| `test_build_testset.py` | Test-set construction |
| `test_check_testset.py` | Test-set integrity checks |
| `test_confusion.py` | Confusion analysis |
| `test_counting.py` | Detection filtering and count aggregation |
| `test_curation_cleaning.py` | Dataset cleaning |
| `test_curation_tagging.py` | Condition tagging |
| `test_dataset.py` | Dataset parsing and mapping |
| `test_defects.py` | Regression / failure-path cases |
| `test_detector.py` | Detector wrappers and registry |
| `test_dev_guard.py` | Development-only access guard |
| `test_evaluation.py` | Evaluation metrics and aggregation |
| `test_expand_testset.py` | Test-set expansion |
| `test_export.py` | JSON/CSV exports |
| `test_features.py` | Label-free feature extraction |
| `test_finetuned.py` | Fine-tuned detector safeguards and evaluation handling |
| `test_history.py` | SQLite history, migrations, review queue and corrections |
| `test_image_cases.py` | Image input edge cases |
| `test_label_app.py` | Blind labelling app |
| `test_predictor.py` | Error predictor |
| `test_prepare_rickshaw_dataset.py` | Rickshaw dataset preparation |
| `test_reliability.py` | Threshold sweep and model agreement |
| `test_reliability_prediction.py` | Validation-gated review flag |
| `test_run_benchmark.py` | Benchmark runner and cache |
| `test_run_week3.py` | Week 3 workflow |
| `test_settings.py` | Settings loading and provenance |
| `test_tag_proposal.py` | Proposed condition tags |
| `test_testset_pipeline.py` | End-to-end test-set pipeline logic |
| `test_tune_on_dev.py` | Development-only tuning |
| `test_validation.py` | Upload validation and decoding |
| `test_video_counting.py` | Video tracking and virtual-line counting |
| `test_zones.py` | Polygon zone analysis |

## Known verification gaps

- Full suite and regression tests: **not run** in this work session.
- Real model-weight integration and first-launch download behaviour: not verified here.
- Interrupted download and offline first launch: remain manual verification items.
- Rapid UI interaction, changing uploads during inference, and repeated fine-tuned-mode
  toggling: require interactive/manual verification in addition to unit tests.
- Streamlit layout at narrow widths, contrast, and font-size checks: not visually verified.
- Video workflow: test coverage exists by filename, but no live video evaluation was run.
- Clean install from `requirements.txt`, headless app launch, and README command-by-command
  walkthrough: pending Part J.
- Benchmark package checksums and image-exclusion verifier: scripts are supplied, but the
  verifier has not been run and the checksum manifest has not yet been generated.
- Usability sessions: not conducted; no participant results or findings are claimed.

## Evidence and interpretation

Test names and existing test-case documentation indicate intended coverage. They are not
a substitute for a fresh run. Model evaluation evidence remains a small pilot on Dhaka
imagery, with human reference counts and the stated pass consistency. The optional
fine-tuned rickshaw mode is a separate model with its own evaluation and contamination
status. Reliability agreement is not correctness. H1 and H2 are not tested; H3 and H4 are
descriptive only.

## Next verification steps (when authorized)

1. Create a clean virtual environment outside the repository.
2. Install `requirements.txt`.
3. Run the full test suite and record exact collection/pass/fail/skip counts.
4. Run `scripts/check_testset.py`.
5. Run the app headlessly and follow the README commands.
6. Run manual defect cases for offline first launch, interrupted downloads, rapid UI
   changes, and repeated optional-mode toggling.
7. Generate and verify `benchmark/SHA256SUMS`.
8. Only then record the actual dependency versions in a constraints file.
