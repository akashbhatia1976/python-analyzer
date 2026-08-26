import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures"
sys.path.insert(0, str(ROOT))

from parameter_structure_router import (  # noqa: E402
    ARRAY,
    FLAT_OBJECT,
    MIXED_OBJECT,
    MIXED_SCALAR_GROUP_OBJECT,
    NESTED_OBJECT,
    PARAMETER_OBJECT_WITH_METADATA,
    UNKNOWN,
    detect_parameter_structure,
    route_parameters,
    NESTED_OBJECT_WITH_SCALAR_GROUPS,
)


def identity_normalizer(name):
    return {
        "originalName": name,
        "canonicalName": name,
        "category": None,
        "normalized": False,
    }


def load(name):
    return json.loads((FIXTURES / name).read_text())


class ParameterRouterTests(unittest.TestCase):
    def test_flat_routes_to_flat_handler(self):
        data = load("flat.json")
        self.assertEqual(detect_parameter_structure(data), FLAT_OBJECT)
        result = route_parameters(data, identity_normalizer)
        self.assertEqual(result["metadata"]["parameterStructure"], FLAT_OBJECT)
        self.assertEqual(len(result["parameters"]), 7)
        self.assertEqual(result["parameters"][0]["name"], "Haemoglobin")
        self.assertEqual(result["parameters"][0]["value"], 13.5)

    def test_array_routes_to_array_handler(self):
        data = load("array.json")
        self.assertEqual(detect_parameter_structure(data), ARRAY)
        result = route_parameters(data, identity_normalizer)
        self.assertEqual(result["metadata"]["parameterStructure"], ARRAY)
        self.assertEqual(len(result["parameters"]), 7)
        self.assertEqual({p["name"] for p in result["parameters"]}, {
            "Haemoglobin", "RBC", "WBC Total Count", "Platelet Count",
            "Creatinine", "Urea", "eGFR"
        })

    def test_nested_routes_to_nested_handler_without_losing_leaves(self):
        data = load("nested.json")
        self.assertEqual(detect_parameter_structure(data), NESTED_OBJECT)
        result = route_parameters(data, identity_normalizer)
        self.assertEqual(result["metadata"]["parameterStructure"], NESTED_OBJECT)
        self.assertEqual(len(result["parameters"]), 7)
        names = {p["name"] for p in result["parameters"]}
        self.assertIn("Haemoglobin", names)
        self.assertIn("Creatinine", names)
        self.assertNotIn("CBC", names)
        self.assertNotIn("Kidney Function Tests", names)

    def test_mixed_routes_to_mixed_handler(self):
        data = load("mixed.json")
        self.assertEqual(detect_parameter_structure(data), MIXED_OBJECT)
        result = route_parameters(data, identity_normalizer)
        self.assertEqual(result["metadata"]["parameterStructure"], MIXED_OBJECT)
        self.assertEqual(len(result["parameters"]), 7)
        self.assertEqual(
            {p["name"] for p in result["parameters"]},
            {"Haemoglobin", "RBC", "WBC Total Count", "Platelet Count", "Creatinine", "Urea", "eGFR"},
        )

    def test_mixed_scalar_group_routes_to_dedicated_handler(self):
        data = load("mixed_scalar_group.json")
        self.assertEqual(
            detect_parameter_structure(data),
            MIXED_SCALAR_GROUP_OBJECT,
        )

        result = route_parameters(data, identity_normalizer)
        self.assertEqual(
            result["metadata"]["parameterStructure"],
            MIXED_SCALAR_GROUP_OBJECT,
        )
        self.assertEqual(len(result["parameters"]), 7)

        by_name = {p["name"]: p for p in result["parameters"]}
        self.assertEqual(by_name["Haemoglobin"]["value"], 6.8)
        self.assertNotIn("sourceGroup", by_name["Haemoglobin"])

        self.assertEqual(by_name["Hypochromia"]["value"], "-")
        self.assertEqual(by_name["Hypochromia"]["sourceGroup"], "WBC Morphology")
        self.assertEqual(by_name["Microcytosis"]["value"], "5")
        self.assertEqual(by_name["Microcytosis"]["sourceGroup"], "WBC Morphology")
        self.assertEqual(by_name["Anisocytosis"]["value"], "Mild")

    def test_qualitative_values_are_preserved(self):
        data = load("qualitative.json")
        self.assertEqual(detect_parameter_structure(data), FLAT_OBJECT)
        result = route_parameters(data, identity_normalizer)
        by_name = {p["name"]: p["value"] for p in result["parameters"]}

        self.assertEqual(by_name["Malarial Parasite"], "Not seen")
        self.assertEqual(by_name["Salmonella typhi 'O'"], "1:80")
        self.assertEqual(by_name["Salmonella typhi 'H'"], "1:40")
        self.assertEqual(by_name["Salmonella paratyphi 'AH'"], "No agglutination")
        self.assertEqual(by_name["Salmonella paratyphi 'BH'"], "No agglutination")
        self.assertEqual(by_name["Dengue NS1 Antigen"], "Negative")

    def test_unknown_structure_fails_safe(self):
        data = load("unknown.json")
        self.assertEqual(detect_parameter_structure(data), UNKNOWN)
        result = route_parameters(data, identity_normalizer)
        self.assertEqual(result["parameters"], [])
        self.assertEqual(result["metadata"]["parameterStructure"], UNKNOWN)
        self.assertEqual(result["metadata"]["status"], "needs_review")

    def test_real_supplied_outputs_remain_flat_and_complete(self):
        expected_counts = {
            "1212_output.json": 43,
            "akash1_output.json": 62,
            "anam1_output.json": 22,
            "cbc_extra_chars_output.json": 7,
        }

        for fixture_name, expected_count in expected_counts.items():
            with self.subTest(fixture=fixture_name):
                payload = load(fixture_name)
                data = payload["extractedParameters"]["Medical Parameters"]
                self.assertEqual(detect_parameter_structure(data), FLAT_OBJECT)
                result = route_parameters(data, identity_normalizer)
                self.assertEqual(len(result["parameters"]), expected_count)



    def test_known_metadata_inside_parameters_routes_to_dedicated_handler(self):
        data = load("parameter_with_metadata.json")

        self.assertEqual(
            detect_parameter_structure(data),
            PARAMETER_OBJECT_WITH_METADATA,
        )

        result = route_parameters(data, identity_normalizer)

        self.assertEqual(
            result["metadata"]["parameterStructure"],
            PARAMETER_OBJECT_WITH_METADATA,
        )

        self.assertEqual(len(result["parameters"]), 2)

        names = {p["name"] for p in result["parameters"]}

        self.assertEqual(
            names,
            {"Haemoglobin", "RBC"},
        )

        self.assertNotIn("Doctor's Notes", names)


    def test_unknown_metadata_is_not_silently_ignored(self):
        data = {
            "CBC": {
                "Haemoglobin": {
                    "Value": 14.7,
                    "Reference Range": "13.0-17.0",
                    "Unit": "g/dL",
                }
            },
            "Something We Have Never Seen Before": [],
        }

        self.assertEqual(
            detect_parameter_structure(data),
            UNKNOWN,
        )

        result = route_parameters(data, identity_normalizer)

        self.assertEqual(result["parameters"], [])
        self.assertEqual(
            result["metadata"]["status"],
            "needs_review",
        )  

    def test_nested_scalar_groups_route_to_dedicated_handler(self):
        data = load("nested_scalar_groups.json")

        self.assertEqual(
            detect_parameter_structure(data),
            NESTED_OBJECT_WITH_SCALAR_GROUPS,
        )

        result = route_parameters(data, identity_normalizer)

        self.assertEqual(
            result["metadata"]["parameterStructure"],
            NESTED_OBJECT_WITH_SCALAR_GROUPS,
        )

        self.assertEqual(len(result["parameters"]), 8)

        by_name = {
            p["name"]: p
            for p in result["parameters"]
        }

        self.assertEqual(
            by_name["Haemoglobin"]["value"],
            6.5,
        )

        self.assertEqual(
            by_name["Platelet Count"]["value"],
            78000,
        )

        self.assertEqual(
            by_name["Hypochromia"]["value"],
            "-",
        )

        self.assertEqual(
            by_name["Hypochromia"]["sourceGroup"],
            "RBC Morphology",
        )

        self.assertEqual(
            by_name["Microcytosis"]["value"],
            5,
        )

        self.assertEqual(
            by_name["Anisocytosis"]["value"],
            "Mild",
        )

        self.assertEqual(
            by_name["Platelet Morphology"]["value"],
            "Platelets reduced on smear.",
        )

        self.assertEqual(
            by_name["Platelet Morphology"]["sourceGroup"],
            "WBC Morphology",
        )

        self.assertEqual(
            by_name["Comment"]["value"],
            "Leucopenia",
        )

        self.assertEqual(
            by_name["ESR"]["value"],
            51,
        )      

if __name__ == "__main__":
    unittest.main(verbosity=2)
