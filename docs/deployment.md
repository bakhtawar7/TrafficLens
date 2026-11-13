# Streamlit Community Cloud demo deployment

## Deployment target

This guide prepares a **public, lightweight image-counting demonstration** on Streamlit
Community Cloud. It is a deployment plan, not evidence that the app has been deployed or
that a hosted runtime has been verified.

- Repository: `bakhtawar7/TrafficLens`
- Working branch reviewed for this guide: `docs/sync-roadmap-status`
- Streamlit entry point: `app/main.py`
- Python dependency file: root `requirements.txt`
- Intended first-use workflow: upload one image in the Image tab and use the default
  pretrained YOLOv8n detector.

Deploying this working branch is optional. If you want the public app to represent the
project's stable release, wait until the branch has been reviewed and intentionally merged
to the release branch. This guide does not merge branches.

## What the first public demo includes

Keep the initial demonstration narrow:

1. Image upload (JPG, JPEG, PNG, or WEBP; application limit 15 MB).
2. Default pretrained YOLOv8n image detection.
3. Per-class and total counts, annotated image, and available JSON/CSV downloads.
4. The existing limitations and research-context notices.

The application has other tabs, but they are not part of the first deployment claim:

- **Reliability:** invoking the comparison workflow loads all three pretrained detectors.
  RT-DETR-L can add substantial model and memory cost. Do not use this as the first
  hosted walkthrough until resource use has been checked.
- **Evaluation:** generated files under `results/` are Git-ignored. A clean deployment
  without those generated artefacts will show the existing planned/placeholder panel,
  not the reported pilot charts.
- **History:** SQLite history is local to the app filesystem. Community Cloud storage
  should not be treated as durable user storage; no external persistent database is
  configured.
- **Video:** the committed video provenance CSV is only a header template, and the video
  path requires local model weights. No video demo data or local video weights are included.
- **Fine-tuned rickshaw mode:** its checkpoint is under `runs/` and is Git-ignored. It is
  not part of the deployable demo. If the toggle is selected without the checkpoint, the
  app reports that the optional model could not run; the default pretrained image result
  remains separate.
- **Usability findings:** no participant sessions have been completed, so do not present
  usability claims.

No curated demo images are currently supplied. Do not use evaluation images as public demo
samples or add images until source, author, licence, and download date are verified.

## Repository preparation status

| Item | Current state | Deployment action |
| --- | --- | --- |
| Streamlit entry point | `app/main.py` exists | Use this exact path |
| Python requirements | Root `requirements.txt` exists | Select it as the dependency source |
| Python version | README target is Python 3.10+; no deployment runtime file | Select Python 3.11 in Advanced settings; confirm compatibility in the hosted build |
| Secrets | No app secrets are required by the inspected code paths | Leave Secrets empty |
| Fine-tuned weights | Not committed; ignored under `runs/` | Exclude from demo |
| Generated evaluation outputs | `results/` is Git-ignored | Expect evaluation placeholder |
| Video samples/weights | No sample video or local video checkpoint supplied | Exclude from demo |
| Persistent database | Local SQLite only | Do not promise durable history |
| Dependency lock | No exact lock or verified constraints file | Do not invent pins; hosted build remains unverified |

The current requirements file uses minimum versions rather than an exact lock and includes
`pytest`, which is not needed by the public image workflow. It is intentionally left
unchanged here: no clean deployment environment or dependency compatibility run has been
performed, and no test-passing versions should be fabricated. A later dependency cleanup
can separate runtime and development requirements after an authorized environment check.

## Deploy on Streamlit Community Cloud

1. Push the reviewed branch to GitHub. The branch currently prepared by this work is
   `docs/sync-roadmap-status`.
2. Sign in at [share.streamlit.io](https://share.streamlit.io/) using the GitHub account
   that can access the repository.
3. Choose **Create app** and select **Yup, I have an app**.
4. Enter:
   - Repository: `bakhtawar7/TrafficLens`
   - Branch: `docs/sync-roadmap-status` (or the approved release branch)
   - Main file path: `app/main.py`
   - Optional app URL: choose an available memorable subdomain.
5. Open **Advanced settings**. Select Python **3.11** as the initial target. This is a
   deployment choice, not a claim that this environment has already been verified. The
   repository currently documents Python 3.10 or newer; the actual hosted build must
   confirm that its dependency floors resolve on the selected interpreter.
6. Leave **Secrets** empty; no secret values are required by the inspected demo code.
7. Deploy and monitor the build logs. If dependency installation fails, record the exact
   package, version, and build error before changing requirements. Do not add speculative
   pins.
8. After the build succeeds, perform the post-deployment checks below before sharing the URL.

Streamlit Community Cloud reads Python dependencies from a root `requirements.txt` (or
from the app directory if one is present). It supports selecting the Python version in
Advanced settings; the documented default is Python 3.12. The chosen version can only be
changed after deployment by deleting and redeploying the app, so choose deliberately.

Official instructions:
- [Deploy an app](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)
- [App dependencies](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies)
- [Manage an app and resource limits](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app)

## Post-deployment verification checklist

These checks are deliberately **not marked complete**. They require a real hosted deployment
and owner approval; none was performed during this documentation preparation.

- [ ] Hosted build completes from the selected branch and Python version.
- [ ] Landing page renders without import or startup errors.
- [ ] A licensed, non-evaluation sample image under 15 MB uploads successfully.
- [ ] Default YOLOv8n weights download on first use and inference completes.
- [ ] Annotated image, class counts, and downloads are correct for the demo sample.
- [ ] A second upload does not expose the previous user's image or result.
- [ ] Evaluation tab clearly remains a placeholder when generated result files are absent.
- [ ] History behavior and data-retention limitations are explained accurately.
- [ ] Reliability, fine-tuned, and video workflows are not represented as verified demo features.
- [ ] Resource use and cold-start behavior are acceptable on the selected hosting tier.
- [ ] Public-facing page explains that model outputs are estimates, not ground truth.

## Privacy, data, and operational cautions

- The app has no authentication layer in the inspected entry point. Treat the Community
  Cloud URL as public to anyone who has it.
- Do not upload private, identifiable, or sensitive road footage to a public demo.
- The app's local history stores metadata and hashes rather than image bytes, but this does
  not make the hosted service a durable or private record system.
- Do not put credentials or private keys in Git. If a future integration needs secrets, add
  them through Community Cloud's Secrets settings and access them with Streamlit's secrets
  interface.
- The default YOLOv8n weights are downloaded on first use by Ultralytics and cached locally.
  The hosted runtime therefore needs outbound access during initial model loading.
- Streamlit Community Cloud resources are limited and may vary. The three-model Reliability
  workflow and large RT-DETR-L model should be treated as higher-risk resource paths until
  observed on the deployed instance.

## Explicitly out of scope for this demo preparation

No test suite, benchmark, application execution, model inference, checksum verifier, or
hosted deployment was run. No model weights, frozen settings, class mappings, cut-offs,
labels, cached detections, evaluation artefacts, or other protected research data were
changed. This document does not certify that the dependencies install or that the app runs
on Community Cloud.
