"""Materials Project API tools for HEA data collection."""

from __future__ import annotations
import json
import os
from datetime import datetime
from pathlib import Path

from config import RAW_DIR, MP_API_KEY, MP_NUM_ELEMENTS_MIN, MP_NUM_ELEMENTS_MAX, MP_MAX_RESULTS


# ── Tool definitions (Anthropic schema format) ────────────────────────────────

MP_TOOL_DEFINITIONS = [
    {
        "name": "query_materials_project",
        "description": (
            "Query the Materials Project database for multi-element alloy compositions. "
            "Retrieves formation energy, band gap, density, bulk modulus, and stability "
            "for compositions containing the specified elements. Best for HEAs with 4-6 elements. "
            "Returns a JSON summary and saves raw data to disk."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "elements": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Chemical element symbols, e.g. ['Al', 'Co', 'Cr', 'Fe', 'Ni']",
                },
                "num_elements_range": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Min and max number of elements [min, max], e.g. [4, 6] for HEAs",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of entries to retrieve (default 500)",
                },
            },
            "required": ["elements"],
        },
    },
]


# ── Tool implementations ──────────────────────────────────────────────────────

def query_materials_project(
    elements: list[str],
    num_elements_range: list[int] | None = None,
    max_results: int = MP_MAX_RESULTS,
) -> str:
    """Query Materials Project and save results to data/raw/."""
    if not MP_API_KEY:
        return "ERROR: MP_API_KEY not set. Get a free key at https://materialsproject.org/api"

    try:
        from mp_api.client import MPRester
    except ImportError:
        return "ERROR: mp-api not installed. Run: pip install mp-api"

    if num_elements_range is None:
        num_elements_range = [MP_NUM_ELEMENTS_MIN, MP_NUM_ELEMENTS_MAX]

    try:
        with MPRester(MP_API_KEY) as mpr:
            results = mpr.materials.summary.search(
                elements=elements,
                num_elements=(num_elements_range[0], num_elements_range[1]),
                fields=[
                    "material_id",
                    "formula_pretty",
                    "formation_energy_per_atom",
                    "energy_above_hull",
                    "density",
                    "band_gap",
                    "is_stable",
                    "bulk_modulus",
                    "shear_modulus",
                ],
            )
    except Exception as e:
        return f"ERROR querying Materials Project: {e}"

    records = []
    for r in results[:max_results]:
        rec = {
            "source": "materials_project",
            "material_id": r.material_id,
            "composition": r.formula_pretty,
            "formation_energy_per_atom": r.formation_energy_per_atom,
            "energy_above_hull": r.energy_above_hull,
            "density": r.density,
            "band_gap": r.band_gap,
            "is_stable": r.is_stable,
        }
        if r.bulk_modulus and hasattr(r.bulk_modulus, "vrh"):
            rec["bulk_modulus_vrh_gpa"] = r.bulk_modulus.vrh
        if r.shear_modulus and hasattr(r.shear_modulus, "vrh"):
            rec["shear_modulus_vrh_gpa"] = r.shear_modulus.vrh
        records.append(rec)

    # Save raw results
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RAW_DIR / f"mp_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(records, f, indent=2, default=str)

    return json.dumps({
        "n_results": len(records),
        "saved_to": str(out_path),
        "elements_queried": elements,
        "sample": records[:3] if records else [],
    }, default=str)


MP_TOOL_CALLABLES = {
    "query_materials_project": query_materials_project,
}
