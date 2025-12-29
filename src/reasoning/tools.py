"""Tools available to the reasoning agent.

Implements tiered web search with snippets-first approach:
- Tier 1 (snippets): Use DuckDuckGo snippets only (~2 sec)
- Tier 2 (selective): Scrape top 2-3 relevant URLs (~5-10 sec)
- Tier 3 (deep): Scrape more URLs for complex research (~15-20 sec)

Based on research from Perplexity architecture, Agentic RAG, and LangChain best practices.

Tools are decorated with @tool for LangChain compatibility, enabling future migration
to native tool binding with models that support it.
"""
import logging
import asyncio
import hashlib
from datetime import datetime, timedelta
from typing import Optional, Literal
import httpx
from bs4 import BeautifulSoup
from ddgs import DDGS
from tenacity import AsyncRetrying, stop_after_attempt, wait_exponential, retry_if_exception_type
from langchain_core.runnables import RunnableConfig
from langchain_core.callbacks import adispatch_custom_event
from langchain_core.tools import tool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# TOOL SCHEMAS (for LangChain bind_tools compatibility)
# =============================================================================

class WebSearchInput(BaseModel):
    """Input schema for web search tool."""

    query: str = Field(
        description="The search query to execute. Be specific and include relevant keywords."
    )
    depth: Literal["snippets", "selective", "deep"] = Field(
        default="snippets",
        description=(
            "Search depth level:\n"
            "- 'snippets': Fast (~2s), uses DuckDuckGo snippets only. Best for simple facts.\n"
            "- 'selective': Medium (~5-10s), scrapes top 3 URLs. Good for detailed info.\n"
            "- 'deep': Slow (~15-20s), scrapes top 7 URLs. For comprehensive research."
        )
    )
    max_results: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of search results to fetch (1-20)."
    )


# =============================================================================
# CACHING
# =============================================================================

_search_cache: dict[str, tuple[str, datetime]] = {}
_CACHE_TTL = timedelta(minutes=10)


def _get_cache_key(query: str, depth: str) -> str:
    """Generate cache key for a search query."""
    return hashlib.md5(f"{query}:{depth}".encode()).hexdigest()


def _get_cached_result(query: str, depth: str) -> Optional[str]:
    """Get cached search result if valid."""
    key = _get_cache_key(query, depth)
    if key in _search_cache:
        result, timestamp = _search_cache[key]
        if datetime.now() - timestamp < _CACHE_TTL:
            logger.info(f"Cache hit for query: {query[:30]}...")
            return result
        else:
            # Expired - remove
            del _search_cache[key]
    return None


def _cache_result(query: str, depth: str, result: str) -> None:
    """Cache a search result."""
    key = _get_cache_key(query, depth)
    _search_cache[key] = (result, datetime.now())

    # Simple cache size limit
    if len(_search_cache) > 100:
        # Remove oldest entries
        oldest_keys = sorted(_search_cache.keys(),
                            key=lambda k: _search_cache[k][1])[:20]
        for k in oldest_keys:
            del _search_cache[k]


# =============================================================================
# SOURCE QUALITY SCORING
# =============================================================================

TRUSTED_DOMAINS = {
    "wikipedia.org": 1.0,
    ".gov": 0.95,
    ".edu": 0.9,
    "reuters.com": 0.9,
    "bbc.com": 0.9,
    "nature.com": 0.95,
    "arxiv.org": 0.9,
    "github.com": 0.85,
    "stackoverflow.com": 0.85,
    "medium.com": 0.6,
    "reddit.com": 0.5,
}


def score_source_quality(url: str) -> float:
    """Score source reliability based on domain."""
    url_lower = url.lower()
    for domain, score in TRUSTED_DOMAINS.items():
        if domain in url_lower:
            return score
    return 0.5  # Unknown source


def score_snippet_relevance(snippet: str, query: str) -> float:
    """Score how relevant a snippet is to the query."""
    if not snippet or not query:
        return 0.0

    query_terms = set(query.lower().split())
    snippet_lower = snippet.lower()

    # Count matching terms
    matches = sum(1 for term in query_terms if term in snippet_lower)
    relevance = matches / len(query_terms) if query_terms else 0.0

    return min(1.0, relevance)


# =============================================================================
# HTML PROCESSING
# =============================================================================

async def fetch_url(client: httpx.AsyncClient, url: str) -> dict:
    """Fetch content from a URL."""
    try:
        response = await client.get(url, timeout=10.0, follow_redirects=True)
        response.raise_for_status()
        return {"url": url, "html": response.text, "error": None}
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 403:
            logger.warning(f"Access denied (403) for {url}. Skipping.")
            return {"url": url, "html": "", "error": "Access Denied (403)"}
        logger.warning(f"Failed to fetch {url}: {e}")
        return {"url": url, "html": "", "error": str(e)}
    except Exception as e:
        logger.warning(f"Failed to fetch {url}: {e}")
        return {"url": url, "html": "", "error": str(e)}


def clean_text(html: str, max_chars: int = 3000) -> str:
    """Extract and clean text from HTML."""
    if not html:
        return ""

    try:
        soup = BeautifulSoup(html, "html.parser")

        # Remove script, style, nav, footer, header elements
        for element in soup(["script", "style", "nav", "footer", "header", "aside", "ads"]):
            element.decompose()

        text = soup.get_text()

        # Collapse whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)

        return text[:max_chars]
    except Exception as e:
        logger.warning(f"HTML parsing failed: {e}")
        return ""


# =============================================================================
# SNIPPET FORMATTING (NO LLM CALLS)
# =============================================================================

def format_snippets(results: list, query: str) -> str:
    """Format search results using snippets (no scraping, no LLM calls).

    This is the fast path - uses DuckDuckGo's body snippets directly.
    """
    if not results:
        return "No search results found."

    formatted = []

    for i, res in enumerate(results, 1):
        title = res.get("title", "No Title")
        url = res.get("href", "")
        snippet = res.get("body", "")  # DuckDuckGo provides this!

        quality = score_source_quality(url)
        relevance = score_snippet_relevance(snippet, query)

        formatted.append(
            f"[{i}] {title}\n"
            f"    URL: {url}\n"
            f"    Quality: {quality:.1f} | Relevance: {relevance:.1f}\n"
            f"    Snippet: {snippet}\n"
        )

    return "\n".join(formatted)


# =============================================================================
# SELECTIVE SCRAPING (NO LLM SUMMARIZATION)
# =============================================================================

async def scrape_top_urls(
    results: list,
    query: str,
    max_urls: int = 3,
    config: RunnableConfig = None
) -> str:
    """Scrape only the top relevant URLs (no LLM summarization).

    The reasoning LLM will synthesize the content itself.
    """
    if not results:
        return ""

    # Score and sort by relevance + quality
    scored_results = []
    for res in results:
        url = res.get("href", "")
        snippet = res.get("body", "")

        quality = score_source_quality(url)
        relevance = score_snippet_relevance(snippet, query)
        combined = (relevance * 0.6) + (quality * 0.4)

        scored_results.append((combined, res))

    scored_results.sort(key=lambda x: x[0], reverse=True)
    top_results = [r[1] for r in scored_results[:max_urls]]

    if config:
        await adispatch_custom_event(
            "tool_io",
            {"type": "selective_scrape", "urls": [r.get("href") for r in top_results]},
            config=config
        )

    # Fetch in parallel
    async with httpx.AsyncClient(headers={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5"
    }) as client:
        tasks = [fetch_url(client, res.get("href")) for res in top_results]
        pages = await asyncio.gather(*tasks)

    # Format scraped content (NO LLM summarization - let reasoning LLM handle it)
    scraped_content = []

    for i, page in enumerate(pages):
        res = top_results[i]
        title = res.get("title", "No Title")
        url = page["url"]

        if page["error"]:
            scraped_content.append(f"[Scraped {i+1}] {title}\nURL: {url}\nError: {page['error']}\n")
            continue

        cleaned = clean_text(page["html"], max_chars=2000)
        if cleaned:
            scraped_content.append(
                f"[Scraped {i+1}] {title}\n"
                f"URL: {url}\n"
                f"Content:\n{cleaned}\n"
            )
        else:
            scraped_content.append(f"[Scraped {i+1}] {title}\nURL: {url}\nNo readable content.\n")

    return "\n---\n".join(scraped_content)


# =============================================================================
# MAIN SEARCH FUNCTION (TIERED)
# =============================================================================

async def perform_web_search(
    query: str,
    depth: str = "snippets",
    max_results: int = 5,
    config: RunnableConfig = None
) -> str:
    """Perform tiered web search with snippets-first approach.

    Args:
        query: The search query string.
        depth: Search depth - "snippets" (fast), "selective" (medium), "deep" (slow)
        max_results: Maximum number of search results to fetch.
        config: RunnableConfig for event dispatch.

    Returns:
        Formatted search results (snippets and/or scraped content).

    Tiers:
        - snippets: Use DuckDuckGo snippets only (~2 sec, 0 LLM calls)
        - selective: Snippets + scrape top 2-3 URLs (~5-10 sec, 0 LLM calls)
        - deep: Snippets + scrape top 5-7 URLs (~15-20 sec, 0 LLM calls)
    """
    logger.info(f"Performing {depth} web search for: {query}")

    # Check cache first
    cached = _get_cached_result(query, depth)
    if cached:
        if config:
            await adispatch_custom_event(
                "tool_io",
                {"type": "cache_hit", "query": query},
                config=config
            )
        return cached

    # Notify start of search
    if config:
        await adispatch_custom_event(
            "tool_io",
            {"type": "search_start", "query": query, "depth": depth},
            config=config
        )

    async for attempt in AsyncRetrying(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True
    ):
        with attempt:
            try:
                # Run DuckDuckGo search in thread (it's synchronous)
                logger.info(f"Running DDGS search (depth={depth})...")

                def run_search():
                    with DDGS() as ddgs:
                        return list(ddgs.text(query, max_results=max_results))

                results = await asyncio.to_thread(run_search)

                if not results:
                    if config:
                        await adispatch_custom_event(
                            "tool_io",
                            {"type": "search_result", "query": query, "count": 0},
                            config=config
                        )
                    return "No search results found."

                # Notify search success
                if config:
                    await adispatch_custom_event(
                        "tool_io",
                        {
                            "type": "search_result",
                            "query": query,
                            "count": len(results),
                            "depth": depth
                        },
                        config=config
                    )

                # === TIER 1: SNIPPETS ONLY (FAST) ===
                snippet_response = format_snippets(results, query)

                if depth == "snippets":
                    _cache_result(query, depth, snippet_response)
                    return snippet_response

                # === TIER 2: SELECTIVE SCRAPING ===
                if depth == "selective":
                    scraped = await scrape_top_urls(results, query, max_urls=3, config=config)
                    combined = f"=== Search Snippets ===\n{snippet_response}\n\n=== Detailed Content ===\n{scraped}"
                    _cache_result(query, depth, combined)
                    return combined

                # === TIER 3: DEEP SCRAPING ===
                if depth == "deep":
                    scraped = await scrape_top_urls(results, query, max_urls=7, config=config)
                    combined = f"=== Search Snippets ===\n{snippet_response}\n\n=== Detailed Content ===\n{scraped}"
                    _cache_result(query, depth, combined)
                    return combined

                # Default fallback
                return snippet_response

            except Exception as e:
                logger.error(f"Search attempt failed: {e}")
                raise


# =============================================================================
# LEGACY COMPATIBILITY
# =============================================================================

async def process_search_results(query: str, results: list, config: RunnableConfig = None) -> str:
    """Legacy function - now uses snippets + selective scraping without LLM summarization."""
    # Format snippets
    snippet_response = format_snippets(results, query)

    # Scrape top 3 for more detail
    scraped = await scrape_top_urls(results, query, max_urls=3, config=config)

    if scraped:
        return f"{snippet_response}\n\n=== Detailed Content ===\n{scraped}"
    return snippet_response


# =============================================================================
# LANGCHAIN TOOL WRAPPERS
# =============================================================================

@tool(args_schema=WebSearchInput)
async def web_search_tool(
    query: str,
    depth: Literal["snippets", "selective", "deep"] = "snippets",
    max_results: int = 5
) -> str:
    """Search the web for information using DuckDuckGo.

    This tool performs tiered web search with different depth levels:
    - snippets: Fast search using DuckDuckGo snippets only (~2 seconds)
    - selective: Medium depth, scrapes top 3 most relevant URLs (~5-10 seconds)
    - deep: Comprehensive search, scrapes top 7 URLs (~15-20 seconds)

    Use this tool when you need to find current information, verify facts,
    or gather data from the web. The results include source URLs and
    quality/relevance scores.

    Always appends the current date to queries for temporal context.

    Args:
        query: The search query. Be specific and include relevant keywords.
        depth: Search depth - "snippets" (fast), "selective" (medium), "deep" (slow).
        max_results: Maximum search results to fetch (1-20).

    Returns:
        Formatted search results with snippets and/or scraped content.
    """
    # Always append current date to all searches for temporal context
    today_date = datetime.now().strftime("%Y-%m-%d")
    query = f"{query} (as of {today_date})"
    logger.info(f"web_search_tool: Search query with date: '{query}'")

    return await perform_web_search(
        query=query,
        depth=depth,
        max_results=max_results,
        config=None  # No config when called via bind_tools
    )


# Export the tool for use with LangChain agents
AVAILABLE_TOOLS = [web_search_tool]
