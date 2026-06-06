from __future__ import annotations

import unittest

import numpy as np

from perception_isp import PerceptionISPConfig, PerceptionISPPipeline, PreviousFrameState, make_synthetic_raw


class PerceptionISPPipelineTest(unittest.TestCase):
    def test_hdr_rggb_pipeline_outputs_dual_path_tensors(self) -> None:
        raw = make_synthetic_raw(width=96, height=64, cfa_pattern="RGGB")
        result = PerceptionISPPipeline().run(raw)
        self.assertEqual(result.vision_rgb.shape, (64, 96, 3))
        self.assertEqual(result.accurate.tensor.shape[:2], (64, 96))
        self.assertEqual(result.accurate.tensor.shape[2], len(result.accurate.channels))
        self.assertEqual(result.fast.tensor.shape[2], len(result.fast.channels))
        self.assertIn("noise_variance", result.maps)
        self.assertIn("snr_map", result.maps)
        self.assertIn("clipping_distance", result.maps)
        self.assertIn("demosaic_confidence", result.maps)
        self.assertIn("edge_confidence", result.maps)
        self.assertIn("hdr_exposure_source", result.maps)
        self.assertIn("clipping_distance", result.accurate.channels)
        self.assertGreaterEqual(result.fast.estimated_latency_us, 0.0)
        self.assertGreater(len(result.accurate.channels), 6)

    def test_supported_cfa_variants_run(self) -> None:
        for cfa in ("RCCB", "RGBIR", "MONO"):
            with self.subTest(cfa=cfa):
                raw = make_synthetic_raw(width=80, height=48, cfa_pattern=cfa)
                result = PerceptionISPPipeline().run(raw)
                self.assertEqual(result.vision_rgb.shape, (48, 80, 3))
                self.assertTrue(np.isfinite(result.accurate.tensor).all())
                self.assertTrue(np.isfinite(result.fast.tensor).all())

    def test_temporal_state_changes_difference_map(self) -> None:
        pipeline = PerceptionISPPipeline()
        first = pipeline.run(make_synthetic_raw(width=80, height=48, frame_counter=0, timestamp_us=0.0))
        second_raw = make_synthetic_raw(width=80, height=48, frame_counter=1, timestamp_us=33333.0)
        second = pipeline.run(second_raw, PreviousFrameState(luma=first.next_state.luma, rgb=first.next_state.rgb, timestamp_us=0.0, frame_counter=0))
        self.assertGreater(float(np.mean(second.maps["temporal_difference"])), 0.0)

    def test_raw_provenance_is_preserved_in_metadata(self) -> None:
        raw = make_synthetic_raw(width=64, height=48)
        raw.provenance = {"bridge": "unit_test", "true_sensor_cfa_mosaic": True}
        result = PerceptionISPPipeline().run(raw)
        self.assertEqual(result.metadata["raw_provenance"]["bridge"], "unit_test")
        self.assertTrue(result.metadata["raw_provenance"]["true_sensor_cfa_mosaic"])

    def test_bayer_demosaic_methods_are_selectable(self) -> None:
        raw = make_synthetic_raw(width=80, height=48, cfa_pattern="RGGB")
        edge_result = PerceptionISPPipeline(PerceptionISPConfig(demosaic_method="edge_aware")).run(raw)
        linear_result = PerceptionISPPipeline(PerceptionISPConfig(demosaic_method="bilinear")).run(raw)
        self.assertEqual(edge_result.metadata["processing"]["demosaic_method"], "edge_aware")
        self.assertEqual(linear_result.metadata["processing"]["demosaic_method"], "bilinear")
        self.assertTrue(np.isfinite(edge_result.vision_rgb).all())
        self.assertTrue(np.isfinite(linear_result.vision_rgb).all())
        self.assertGreater(float(np.mean(np.abs(edge_result.vision_rgb - linear_result.vision_rgb))), 1.0e-5)

    def test_detector_log_tone_mapping_matches_gamma_encoded_log(self) -> None:
        raw = make_synthetic_raw(width=80, height=48, cfa_pattern="RGGB")
        detector_result = PerceptionISPPipeline(PerceptionISPConfig(tone_mapping="detector_log")).run(raw)
        human_log_result = PerceptionISPPipeline(PerceptionISPConfig(tone_mapping="human_log")).run(raw)
        self.assertEqual(detector_result.metadata["processing"]["tone_mapping"], "detector_log")
        self.assertTrue(np.allclose(detector_result.vision_rgb, human_log_result.vision_rgb))


if __name__ == "__main__":
    unittest.main()
