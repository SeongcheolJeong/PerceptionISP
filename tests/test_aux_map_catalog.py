from __future__ import annotations

import unittest

from perception_isp.core.aux_map_catalog import (
    AUX_MAP_SPECS,
    aux_map_catalog_json,
    validate_aux_map_catalog,
)
from perception_isp.core.aux_map_rationale import (
    AUX_MAP_RATIONALES,
    aux_map_rationale_json,
    validate_aux_map_rationales,
)
from perception_isp.core.pipeline import PerceptionISPPipeline
from perception_isp.core.synthetic import make_synthetic_raw


class AuxMapCatalogTest(unittest.TestCase):
    def test_catalog_is_complete_and_explanatory(self) -> None:
        result = PerceptionISPPipeline().run(make_synthetic_raw(width=32, height=24))
        validate_aux_map_catalog(result.maps)
        self.assertEqual(len(AUX_MAP_SPECS), 33)
        self.assertEqual({row.name for row in AUX_MAP_SPECS}, set(result.maps))
        for row in aux_map_catalog_json():
            with self.subTest(name=row["name"]):
                for field in (
                    "purpose",
                    "expected_effect",
                    "algorithm",
                    "value_semantics",
                    "applicability",
                    "limitations",
                    "implementation_ref",
                ):
                    self.assertTrue(row[field].strip())

    def test_catalog_rejects_code_drift(self) -> None:
        with self.assertRaisesRegex(ValueError, "uncatalogued"):
            validate_aux_map_catalog([*(row.name for row in AUX_MAP_SPECS), "new_map"])

    def test_rationale_catalog_covers_every_map_with_formula_and_boundary(self) -> None:
        validate_aux_map_rationales()
        self.assertEqual(len(AUX_MAP_RATIONALES), 33)
        self.assertEqual(
            {row.name for row in AUX_MAP_RATIONALES},
            {row.name for row in AUX_MAP_SPECS},
        )
        for row in aux_map_rationale_json():
            with self.subTest(name=row["name"]):
                for field in (
                    "problem_situation",
                    "formula",
                    "why_it_helps",
                    "design_basis",
                    "interpretation_boundary",
                ):
                    self.assertTrue(row[field].strip())

    def test_rationale_catalog_rejects_public_contract_drift(self) -> None:
        with self.assertRaisesRegex(ValueError, "rationale catalog drift"):
            validate_aux_map_rationales(
                [*(row.name for row in AUX_MAP_SPECS), "undocumented_map"]
            )


if __name__ == "__main__":
    unittest.main()
