# Migration from v0.1 to v0.2

v0.2 is a deliberate breaking reorganization. The flat package and multiple
console commands were replaced by functional subpackages and one CLI.

## Install and Python

- Minimum Python changed from 3.9 to 3.11.
- Install profiles are now `raw`, `ml`, `camerae2e`, `dev`, and `all`.
- Reinstall editable environments after pulling v0.2:

```bash
python -m pip install -e '.[dev,raw,ml,camerae2e]'
```

## Python Imports

| v0.1 | v0.2 |
| --- | --- |
| `perception_isp.pipeline` | `perception_isp.core.pipeline` |
| `perception_isp.types` | `perception_isp.core.types` |
| `perception_isp.aux_export` | `perception_isp.datasets.aux_export` |
| `perception_isp.kitti_dataset` | `perception_isp.datasets.kitti_dataset` |
| `perception_isp.pascalraw_loader` | `perception_isp.datasets.pascalraw_loader` |
| `perception_isp.yolo_aux_dataset` | `perception_isp.datasets.yolo_aux_dataset` |
| `perception_isp.yolo_aux_train` | `perception_isp.training.yolo_aux_train` |
| `perception_isp.eval_cli` | `perception_isp.evaluation.eval_cli` |
| `perception_isp.claim_dashboard` | `perception_isp.reporting.claim_dashboard` |

Only the main pipeline/config/result types are exported from the package root.
Internal tools should import their functional subpackage explicitly.

## CLI Commands

| v0.1 command | v0.2 command |
| --- | --- |
| `perception-isp` | `perception-isp isp run` |
| `perception-isp-eval` | `perception-isp evaluate detection` |
| `perception-isp-sweep` | `perception-isp evaluate resolution` |
| `perception-isp-aux-export` | `perception-isp aux export` |
| `perception-isp-aux-train-smoke` | `perception-isp train smoke` |
| `perception-isp-aux-train-dense` | `perception-isp train dense` |
| `perception-isp-aux-eval-dense` | `perception-isp evaluate dense` |
| `perception-isp-report-rollup` | `perception-isp report rollup` |
| `perception-isp-prepare-coco-subset` | `perception-isp data coco-subset` |
| `perception-isp-merge-comparison-reports` | `perception-isp report merge` |
| `perception-isp-isp-sweep` | `perception-isp evaluate isp-sweep` |
| `perception-isp-threshold-sweep` | `perception-isp evaluate threshold` |
| `perception-isp-proposal-calibration` | `perception-isp evaluate calibrate` |
| `perception-isp-apply-proposal-calibration` | `perception-isp evaluate apply-calibration` |
| `perception-isp-scene-edge-casebook` | `perception-isp report edge-casebook` |
| `perception-isp-scene-edge-aux-sweep` | `perception-isp evaluate edge-aux` |
| `perception-isp-yolo-aux-dataset` | `perception-isp data yolo-aux` |
| `perception-isp-yolo-aux-train` | `perception-isp train yolo-aux` |

The options after each command remain owned by the same workflow parser.

## Environment Paths

Replace personal absolute paths with:

```bash
export PERCEPTION_ISP_DATA=/path/to/data
export PERCEPTION_ISP_OUTPUT=/path/to/results
export CAMERAE2E_ROOT=/path/to/CameraE2E
```

## Checkpoints

The v0.2 gated-stem module registers the v0.1
`perception_isp.yolo_aux_train` pickle alias before loading YOLO checkpoints.
Load old checkpoints through `perception-isp train yolo-aux` or an evaluation
module that imports the v0.2 gated stem first. After a successful load, save a
new checkpoint so future runs use the v0.2 module path.

## Historical Commands

Files under `docs/history` preserve old commands and result paths for
traceability. They are not updated operating instructions. Use
`USER_GUIDE_KO.md` for current commands.
