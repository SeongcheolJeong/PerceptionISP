# Executable Examples

The examples call the same implementation as `perception-isp example suite`.
They generate an English HTML report, a machine-readable `summary.json`, and
PNG assets under `${PERCEPTION_ISP_OUTPUT:-reports}`.

Run the complete self-contained suite:

```bash
python examples/run_example_suite.py
```

Run one focused example:

```bash
python examples/hdr_example.py
python examples/sensor_metadata_example.py
python examples/calibration_optics_example.py
python examples/temporal_example.py
python examples/dnn_contract_example.py
```

The default suite does not require external datasets, PyTorch, or CameraE2E.
To require the real CameraE2E bridge check, configure the local checkout and
run:

```bash
export CAMERAE2E_ROOT=/path/to/CameraE2E
python -m pip install -e "$CAMERAE2E_ROOT"
python -m pip install -e '.[camerae2e]'
perception-isp example suite --with-camerae2e --scene 'uniform ee'
```

`PASS` means that a controlled front-end mechanism behaved in the expected
direction. It is not a held-out detector or segmentation performance claim.
Scalar sensor metadata is shown in the report but is not part of the current
RGB+Aux DNN tensor.
