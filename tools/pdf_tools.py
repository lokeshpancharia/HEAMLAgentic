"""PDF parsing tools: extract HEA composition-property data from papers."""

from __future__ import annotations
import json
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path

import requests

from config import PAPERS_DIR, RAW_DIR


PDF_TOOL_DEFINITIONS = [
    {
        "name": "extract_data_from_pdf",
        "description": (
            "Download and parse a PDF (from URL or local path). Uses PyMuPDF to extract "
            "text, then uses an LLM to identify and extract composition-property data tables. "
            "Returns extracted HEA records as JSON. Use this after search_arxiv_papers to "
            "extract quantitative data from each relevant paper."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "source": {
                    "type": "string",
                    "description": "URL (https://arxiv.org/pdf/...) or local file path to PDF",
                },
                "target_property": {
                    "type": "string",
                    "description": "Property to extract, e.g. 'hardness', 'yield_strength'",
                },
                "paper_title": {
                    "type": "string",
                    "description": "Optional paper title for provenance tracking",
                },
            },
            "required": ["source"],
        },
    },
]


def _download_pdf(url: str) -> Path | None:
    """Download a PDF from URL into the papers directory."""
    try:
        resp = requests.get(url, timeout=30, headers={"User-Agent": "HEAMLAgentic/1.0"})
        resp.raise_for_status()
        fname = url.split("/")[-1]
        if not fname.endswith(".pdf"):
            fname += ".pdf"
        path = PAPERS_DIR / fname
        with open(path, "wb") as f:
            f.write(resp.content)
        return path
    except Exception:
        return None


def _extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract full text from PDF using PyMuPDF."""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        return ""

    text = []
    try:
        with fitz.open(str(pdf_path)) as doc:
            for page in doc:
                text.append(page.get_text())
    except Exception:
        pass
    return "\n".join(text)


def _llm_extract_data(text: str, target_property: str, paper_title: str) -> list[dict]:
    """Use Anthropic Claude to extract structured data from paper text."""
    try:
        import anthropic
    except ImportError:
        return []

    api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        return []

    # Truncate text to fit in context (use first 8000 chars which usually has tables)
    truncated = text[:8000]

    prompt = f"""You are a materials science data extractor. Extract ALL composition-property data from this research paper text.

Paper: {paper_title or 'Unknown'}
Target property: {target_property or 'any mechanical or physical property'}

Text:
{truncated}

Extract every row of data that contains:
1. A chemical composition (e.g., Al0.5CoCrFeNi, CrMnFeCoNi, etc.)
2. At least one measured property value

Return ONLY a JSON array (no other text) with objects having these keys:
- "composition": string (e.g., "Al0.5CoCrFeNi")
- "property_name": string (e.g., "hardness", "yield_strength")
- "value": number
- "unit": string (e.g., "HV", "MPa", "GPa")
- "condition": string (e.g., "as-cast", "annealed 1000C")

If no data is found, return [].
Only return the JSON array, nothing else."""

    try:
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY", ""))
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()
        # Clean JSON
        raw = re.sub(r"^```json\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        return json.loads(raw)
    except Exception:
        return []


def extract_data_from_pdf(
    source: str,
    target_property: str = "",
    paper_title: str = "",
) -> str:
    """Main tool: download PDF, extract text, run LLM extraction."""
    # Determine path
    if source.startswith("http"):
        pdf_path = _download_pdf(source)
        if pdf_path is None:
            return json.dumps({"error": f"Failed to download PDF from {source}", "records": []})
    else:
        pdf_path = Path(source)
        if not pdf_path.exists():
            return json.dumps({"error": f"File not found: {source}", "records": []})

    # Extract text
    text = _extract_text_from_pdf(pdf_path)
    if not text:
        return json.dumps({
            "error": "Failed to extract text from PDF (may be image-based)",
            "pdf_path": str(pdf_path),
            "records": [],
        })

    # LLM extraction
    records = _llm_extract_data(text, target_property, paper_title)

    # Add provenance
    for rec in records:
        rec["source"] = "paper_pdf"
        rec["paper"] = paper_title or str(pdf_path.name)
        rec["pdf_url"] = source

    # Save extracted data
    if records:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = RAW_DIR / f"pdf_extract_{ts}.json"
        with open(out_path, "w") as f:
            json.dump(records, f, indent=2)
        saved_to = str(out_path)
    else:
        saved_to = None

    return json.dumps({
        "n_records": len(records),
        "pdf_path": str(pdf_path),
        "saved_to": saved_to,
        "records": records[:5],  # preview only
    }, default=str)


PDF_TOOL_CALLABLES = {
    "extract_data_from_pdf": extract_data_from_pdf,
}
