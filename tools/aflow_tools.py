"""AFLOW materials database tools using AFLUX REST API."""

from __future__ import annotations
import json
import requests
from datetime import datetime

from config import RAW_DIR, AFLOW_MAX_RESULTS


AFLOW_TOOL_DEFINITIONS = [
    {
        "name": "query_aflow",
        "description": (
            "Query the AFLOW materials database using the AFLUX REST API. "
            "Good for structural, thermodynamic, and electronic property data on alloys. "
            "Returns composition, formation enthalpy, band gap, density, and more."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "species": {
                    "type": "string",
                    "description": "Comma-separated element symbols, e.g. 'Al,Co,Cr,Fe,Ni'",
                },
                "properties": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "AFLOW keywords to retrieve. Common: "
                        "['enthalpy_formation_atom', 'Egap', 'density', 'volume_atom', 'compound']"
                    ),
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum entries to retrieve (default 200)",
                },
            },
            "required": ["species"],
        },
    },
]


def query_aflow(
    species: str,
    properties: list[str] | None = None,
    max_results: int = AFLOW_MAX_RESULTS,
) -> str:
    """Query AFLOW via AFLUX REST API."""
    if properties is None:
        properties = ["compound", "enthalpy_formation_atom", "Egap", "density",
                      "volume_atom", "nspecies", "natoms"]

    # Always include compound
    if "compound" not in properties:
        properties = ["compound"] + properties

    # AFLUX query: species match filter
    species_list = [s.strip() for s in species.split(",")]
    species_filter = ",".join(f"species({sp})" for sp in species_list)
    select_fields = ",".join(properties)

    # AFLUX URL
    url = (
        f"http://aflow.org/API/aflux/?"
        f"{species_filter},"
        f"nspecies({len(species_list)}),"
        f"paging(1,{min(max_results, 1000)}),"
        f"format(json),"
        f"{select_fields}"
    )

    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        return f"ERROR querying AFLOW: {e}"
    except json.JSONDecodeError:
        return "ERROR: AFLOW returned non-JSON response"

    if not isinstance(data, list):
        data = [data] if isinstance(data, dict) else []

    # Normalize records
    records = []
    for entry in data[:max_results]:
        rec = {"source": "aflow"}
        for prop in properties:
            rec[prop] = entry.get(prop)
        # Rename compound → composition for consistency
        if "compound" in rec:
            rec["composition"] = rec.pop("compound")
        records.append(rec)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RAW_DIR / f"aflow_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(records, f, indent=2, default=str)

    return json.dumps({
        "n_results": len(records),
        "saved_to": str(out_path),
        "species_queried": species_list,
        "properties_retrieved": properties,
        "sample": records[:3] if records else [],
    }, default=str)


AFLOW_TOOL_CALLABLES = {
    "query_aflow": query_aflow,
}
