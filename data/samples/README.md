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

A small public Streamlit image-counting demo does not require this curated sample folder;
viewers can upload their own non-sensitive, appropriately licensed images. Do not run the
warm-cache script as part of deployment preparation: it targets all pretrained registry
models and the separate fine-tuned rickshaw model, not just the lightweight YOLOv8n demo
path. Any future cache-warming run requires approved samples and separate owner approval.
