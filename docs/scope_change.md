# Scope change: optional fine-tuned rickshaw mode

The project began with a pretrained-only counting scope. It now includes an optional,
off-by-default, one-class fine-tuned rickshaw detector in the Image tab. Standard
pretrained counts, ranges, reliability agreements, and flags remain separate and
unchanged by this mode.

Mentor approval text: **not yet obtained**.

The fine-tuned detector is experimental, does not detect other vehicle categories,
and has unresolved training-data licence and citation metadata. Its pilot results and
contamination exclusions are documented in `rickshaw_model.md`.
