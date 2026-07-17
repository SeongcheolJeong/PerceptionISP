# Published PerceptionISP report snapshots

This directory intentionally checks in a small, reviewable subset of the local
report artifacts:

- [Project purpose, architecture, status, and roadmap](perception_project_status_current_v1/index.html)
- [nuScenes → CameraE2E → PerceptionISP pipeline showcase](perception_nuscenes_scene0061_camerae2e_pipeline_showcase_v1/index.html)
- [Historical accomplishment report](perception_project_accomplishment_tabs_current_v1/index.html)

The showcase is a **preview-only publication bundle**. It contains HTML,
machine-readable metadata, and 427 PNG previews, but omits the 976 lossless NPY
arrays from the approximately 1.1 GB local acceptance bundle. The original
nuScenes JPEGs and source dataset tables are not included.

The nuScenes-derived PNG previews are governed by the report-local
[data and attribution notice](perception_nuscenes_scene0061_camerae2e_pipeline_showcase_v1/DATA_NOTICE.md),
CC BY-NC-SA 4.0, and the additional nuScenes Dataset Terms. Code and other
repository content are not relicensed by that data notice.

`artifacts_manifest.json` and `checksums.sha256` describe only the checked-in
preview bundle. Regenerate the full numerical acceptance bundle from a local
nuScenes checkout when lossless array review is required.
