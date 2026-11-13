# Demo samples

No image samples are currently included. The folder is intentionally left empty until each
candidate image has a verified source, author, licence, and download date. Do not copy
evaluation images into the demo set or use unlicensed images.

Add 8–10 approved files after provenance review, including:
- development examples first;
- an easy daytime road scene;
- a dense traffic scene;
- a low-light scene;
- a scene with visible rickshaws;
- one deliberate failure case.

Maintain a manifest with filename, source URL, author, licence text, download date,
selection reason, and whether the image is from the development or a separate demo pool.
Do not include evaluation images or expose per-image evaluation model counts in a demo.

Once the sample set exists, run `python scripts/warm_cache.py` before the demo. The script
warms all pretrained registry models and the separate fine-tuned rickshaw model. It does not
change settings or evaluation artefacts.
