"""arXiv paper search tools for HEA literature mining."""

from __future__ import annotations
import json

from config import ARXIV_CATEGORIES, ARXIV_MAX_RESULTS


ARXIV_TOOL_DEFINITIONS = [
    {
        "name": "search_arxiv_papers",
        "description": (
            "Search arXiv for HEA research papers. Returns paper metadata including "
            "title, authors, abstract, arxiv_id, and pdf_url. Use the PDF URLs with "
            "extract_data_from_pdf to extract composition-property data."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Search query, e.g. 'high entropy alloy hardness machine learning' "
                        "or 'multi-principal element alloy yield strength'"
                    ),
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum papers to return (default 20)",
                },
                "categories": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "arXiv categories filter, e.g. ['cond-mat.mtrl-sci']",
                },
            },
            "required": ["query"],
        },
    },
]


def search_arxiv_papers(
    query: str,
    max_results: int = ARXIV_MAX_RESULTS,
    categories: list[str] | None = None,
) -> str:
    """Search arXiv and return paper metadata."""
    try:
        import arxiv
    except ImportError:
        return "ERROR: arxiv package not installed. Run: pip install arxiv"

    if categories is None:
        categories = ARXIV_CATEGORIES

    # Build category filter
    if categories:
        cat_filter = " OR ".join(f"cat:{c}" for c in categories)
        full_query = f"({query}) AND ({cat_filter})"
    else:
        full_query = query

    try:
        client = arxiv.Client()
        search = arxiv.Search(
            query=full_query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        papers = []
        for result in client.results(search):
            papers.append({
                "arxiv_id": result.entry_id.split("/")[-1],
                "title": result.title,
                "authors": [a.name for a in result.authors[:5]],
                "abstract": result.summary[:500] + "..." if len(result.summary) > 500 else result.summary,
                "published": str(result.published.date()),
                "pdf_url": result.pdf_url,
                "categories": result.categories,
            })
    except Exception as e:
        return f"ERROR searching arXiv: {e}"

    return json.dumps({
        "n_papers": len(papers),
        "query_used": full_query,
        "papers": papers,
    }, default=str)


ARXIV_TOOL_CALLABLES = {
    "search_arxiv_papers": search_arxiv_papers,
}
