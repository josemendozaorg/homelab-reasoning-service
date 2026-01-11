"""Tools available to the reasoning agent.

Implements tiered web search with support for multiple providers:
- Tavily: Best for agents (clean content, no scraping needed)
- Brave: High privacy, large independent index
- Google: Best coverage (requires API key + CSE ID)
- DuckDuckGo: Free fallback (snippets + scraping)

Based on research from Perplexity architecture, Agentic RAG, and LangChain best practices.
"""
import logging
import asyncio
import hashlib
import json
from datetime import datetime, timedelta
from typing import Optional, Literal
import httpx
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
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
            "- 'snippets': Fast (~2s), uses snippets only. Best for simple facts.\n"
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


def _get_cache_key(query: str, depth: str, provider: str) -> str:
    """Generate cache key for a search query."""
    return hashlib.md5(f"{provider}:{query}:{depth}".encode()).hexdigest()


def _get_cached_result(query: str, depth: str, provider: str) -> Optional[str]:
    """Get cached search result if valid."""
    key = _get_cache_key(query, depth, provider)
    if key in _search_cache:
        result, timestamp = _search_cache[key]
        if datetime.now() - timestamp < _CACHE_TTL:
            logger.info(f"Cache hit for query: {query[:30]}...")
            return result
        else:
            # Expired - remove
            del _search_cache[key]
    return None


def _cache_result(query: str, depth: str, provider: str, result: str) -> None:
    """Cache a search result."""
    key = _get_cache_key(query, depth, provider)
    _search_cache[key] = (result, datetime.now())

    # Simple cache size limit
    if len(_search_cache) > 100:
        # Remove oldest entries
        oldest_keys = sorted(_search_cache.keys(),
                            key=lambda k: _search_cache[k][1])[:20]
        for k in oldest_keys:
            del _search_cache[k]


# =============================================================================
# HTML PROCESSING & SCRAPING
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


async def scrape_top_urls(
    results: list,
    query: str,
    max_urls: int = 3,
    config: RunnableConfig = None
) -> str:
    """Scrape only the top relevant URLs."""
    if not results:
        return ""

    # Simple heuristic: take top N results
    # (Assuming results are already ranked by search engine)
    top_results = results[:max_urls]

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

    # Format scraped content
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
# SEARCH PROVIDERS
# =============================================================================

async def search_duckduckgo(query: str, max_results: int) -> list[dict]:
    """Search using DuckDuckGo (No API Key)."""
    def run_search():
        with DDGS() as ddgs:
            return list(ddgs.text(query, max_results=max_results))

    results = await asyncio.to_thread(run_search)
    # Normalize keys
    return [{"title": r.get("title"), "href": r.get("href"), "body": r.get("body")} for r in results]


async def search_tavily(query: str, max_results: int, api_key: str, depth: str = "snippets") -> dict:
    """Search using Tavily API.

    Tavily returns parsed content, so we can often skip manual scraping.
    """
    if not api_key:
        raise ValueError("Tavily API Key is missing.")

    # Tavily 'search_depth' parameter
    tavily_depth = "advanced" if depth in ["selective", "deep"] else "basic"
    include_raw_content = depth == "deep"

    url = "https://api.tavily.com/search"
    payload = {
        "api_key": api_key,
        "query": query,
        "search_depth": tavily_depth,
        "include_answer": True,
        "include_raw_content": include_raw_content,
        "max_results": max_results
    }

    async with httpx.AsyncClient() as client:
        resp = await client.post(url, json=payload, timeout=15.0)
        resp.raise_for_status()
        data = resp.json()

    # Tavily returns a rich response. We can return it directly formatted.
    results = data.get("results", [])
    answer = data.get("answer", "")

    formatted_results = []

    # Format Answer first if available
    if answer:
        formatted_results.append(f"=== AI Overview (Tavily) ===\n{answer}\n")

    formatted_results.append("=== Search Results ===")

    for i, res in enumerate(results, 1):
        title = res.get("title", "No Title")
        url = res.get("url", "")
        content = res.get("content", "")
        raw = res.get("raw_content", "")

        # Prefer content, fall back to snippet
        body = content if len(content) > 50 else res.get("snippet", "")

        formatted_results.append(
            f"[{i}] {title}\n"
            f"    URL: {url}\n"
            f"    Content: {body}\n"
        )

        if raw and depth == "deep":
             formatted_results.append(f"    Raw (Partial): {raw[:500]}...\n")

    return "\n".join(formatted_results)


async def search_brave(query: str, max_results: int, api_key: str) -> list[dict]:
    """Search using Brave Search API."""
    if not api_key:
        raise ValueError("Brave API Key is missing.")

    url = "https://api.search.brave.com/res/v1/web/search"
    headers = {
        "X-Subscription-Token": api_key,
        "Accept": "application/json"
    }
    params = {
        "q": query,
        "count": min(max_results, 20)
    }

    async with httpx.AsyncClient() as client:
        resp = await client.get(url, headers=headers, params=params, timeout=10.0)
        resp.raise_for_status()
        data = resp.json()

    # Brave results are in data['web']['results']
    results = data.get("web", {}).get("results", [])

    # Normalize to standard format
    return [
        {
            "title": r.get("title"),
            "href": r.get("url"),
            "body": r.get("description")
        }
        for r in results
    ]


async def search_google(query: str, max_results: int, api_key: str, cse_id: str) -> list[dict]:
    """Search using Google Custom Search JSON API."""
    if not api_key or not cse_id:
        raise ValueError("Google API Key or CSE ID is missing.")

    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": api_key,
        "cx": cse_id,
        "q": query,
        "num": min(max_results, 10) # Google caps at 10 per request
    }

    async with httpx.AsyncClient() as client:
        resp = await client.get(url, params=params, timeout=10.0)
        resp.raise_for_status()
        data = resp.json()

    results = data.get("items", [])

    # Normalize
    return [
        {
            "title": r.get("title"),
            "href": r.get("link"),
            "body": r.get("snippet")
        }
        for r in results
    ]


# =============================================================================
# MAIN SEARCH FUNCTION (DISPATCHER)
# =============================================================================

async def perform_web_search(
    query: str,
    depth: str = "snippets",
    max_results: int = 5,
    config: RunnableConfig = None
) -> str:
    """Perform tiered web search with the configured provider.

    Tiers:
        - snippets: Provider API only.
        - selective: Provider API + Scrape top 3 URLs (unless Tavily).
        - deep: Provider API + Scrape top 7 URLs (unless Tavily).
    """
    # 1. Extract Configuration
    configuration = config.get("configurable", {}) if config else {}
    provider = configuration.get("search_provider", "ddg")
    api_key = configuration.get("search_api_key")
    cse_id = configuration.get("search_cse_id")

    logger.info(f"Performing {depth} web search for: '{query}' using {provider}")

    # 2. Check Cache
    cached = _get_cached_result(query, depth, provider)
    if cached:
        if config:
            await adispatch_custom_event(
                "tool_io",
                {"type": "cache_hit", "query": query, "provider": provider},
                config=config
            )
        return cached

    # 3. Notify start
    if config:
        await adispatch_custom_event(
            "tool_io",
            {"type": "search_start", "query": query, "depth": depth, "provider": provider},
            config=config
        )

    # 4. Execute Search
    async for attempt in AsyncRetrying(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True
    ):
        with attempt:
            try:
                # --- STRATEGY: TAVILY (Special Case: Handles scraping internally) ---
                if provider == "tavily":
                    result = await search_tavily(query, max_results, api_key, depth)
                    _cache_result(query, depth, provider, result)
                    return result

                # --- STRATEGY: OTHERS (Fetch Snippets -> Optional Scrape) ---
                results = []
                if provider == "brave":
                    results = await search_brave(query, max_results, api_key)
                elif provider == "google":
                    results = await search_google(query, max_results, api_key, cse_id)
                else: # Default to DDG
                    results = await search_duckduckgo(query, max_results)

                if not results:
                    return "No search results found."

                # Notify success (count)
                if config:
                    await adispatch_custom_event(
                        "tool_io",
                        {"type": "search_result", "count": len(results)},
                        config=config
                    )

                # Format Snippets
                formatted_snippets = []
                for i, res in enumerate(results, 1):
                    formatted_snippets.append(
                        f"[{i}] {res['title']}\n"
                        f"    URL: {res['href']}\n"
                        f"    Snippet: {res['body']}\n"
                    )
                snippet_text = "\n".join(formatted_snippets)

                # Return if snippets only
                if depth == "snippets":
                    _cache_result(query, depth, provider, snippet_text)
                    return snippet_text

                # Perform Scraping for Selective/Deep
                max_urls = 3 if depth == "selective" else 7
                scraped = await scrape_top_urls(results, query, max_urls=max_urls, config=config)

                combined = f"=== Search Snippets ===\n{snippet_text}\n\n=== Detailed Content ===\n{scraped}"
                _cache_result(query, depth, provider, combined)
                return combined

            except Exception as e:
                logger.error(f"Search attempt failed ({provider}): {e}")

                # Fallback to DDG if primary fails?
                if provider != "ddg":
                    logger.warning("Falling back to DuckDuckGo...")
                    provider = "ddg"
                    # The loop will retry with provider="ddg" because we modified the local var?
                    # No, AsyncRetrying retries the whole block.
                    # We need to manually trigger fallback logic or just let it fail.
                    # For simplicity, we just raise and let the user/agent handle failure,
                    # OR we can recursively call perform_web_search with ddg.
                    # Let's just raise for now to avoid complex recursion depth issues.
                    raise e
                raise


# =============================================================================
# LANGCHAIN TOOL WRAPPERS
# =============================================================================

@tool(args_schema=WebSearchInput)
async def web_search_tool(
    query: str,
    depth: Literal["snippets", "selective", "deep"] = "snippets",
    max_results: int = 5,
    config: RunnableConfig = None
) -> str:
    """Search the web for information using the configured provider.

    This tool performs tiered web search with different depth levels:
    - snippets: Fast search using search engine snippets only.
    - selective: Medium depth, scrapes top 3 most relevant URLs.
    - deep: Comprehensive search, scrapes top 7 URLs (or uses deep API mode).

    Use this tool when you need to find current information, verify facts,
    or gather data from the web. The results include source URLs.

    Args:
        query: The search query. Be specific and include relevant keywords.
        depth: Search depth - "snippets" (fast), "selective" (medium), "deep" (slow).
        max_results: Maximum search results to fetch (1-20).

    Returns:
        Formatted search results with snippets and/or scraped content.
    """
    # Always append current date to all searches for temporal context
    today_date = datetime.now().strftime("%Y-%m-%d")
    query_with_date = f"{query} (as of {today_date})"

    # Config is injected by LangChain at runtime
    return await perform_web_search(
        query=query_with_date,
        depth=depth,
        max_results=max_results,
        config=config
    )

# Export the tool for use with LangChain agents
AVAILABLE_TOOLS = [web_search_tool]
