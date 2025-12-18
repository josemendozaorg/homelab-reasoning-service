"""Tools available to the reasoning agent."""
import logging
import asyncio
import httpx
from bs4 import BeautifulSoup
from ddgs import DDGS
import dspy
from tenacity import AsyncRetrying, stop_after_attempt, wait_exponential, retry_if_exception_type
from .signatures import SummarizeSignature

logger = logging.getLogger(__name__)

async def fetch_url(client: httpx.AsyncClient, url: str) -> str:
    """Fetch content from a URL."""
    try:
        response = await client.get(url, timeout=10.0, follow_redirects=True)
        response.raise_for_status()
        return response.text
    except Exception as e:
        logger.warning(f"Failed to fetch {url}: {e}")
        return ""

def clean_text(html: str) -> str:
    """Extract and clean text from HTML."""
    if not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")
    
    # Remove script and style elements
    for script in soup(["script", "style", "nav", "footer", "header"]):
        script.decompose()
        
    text = soup.get_text()
    
    # Collapse whitespace
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = '\n'.join(chunk for chunk in chunks if chunk)
    
    return text[:10000]  # Limit context per page

async def process_search_results(query: str, results: list) -> str:
    """Scrape and summarize search results in parallel."""
    async with httpx.AsyncClient(headers={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5"
    }) as client:
        tasks = []
        for res in results:
            url = res.get('href')
            if url:
                tasks.append(fetch_url(client, url))
        
        # Parallel fetch
        pages_content = await asyncio.gather(*tasks)

    summaries = []
    summarizer = dspy.Predict(SummarizeSignature)
    
    for res, content in zip(results, pages_content):
        title = res.get('title', 'No Title')
        url = res.get('href', '#')
        
        if not content:
            summaries.append(f"Source: {title} ({url})\nFailed to retrieve content.\n")
            continue
            
        cleaned_content = clean_text(content)
        if not cleaned_content:
            summaries.append(f"Source: {title} ({url})\nNo readable content found.\n")
            continue
            
        # Summarize with DSPy
        try:
            # Run summarization in thread to avoid blocking loop
            pred = await asyncio.to_thread(summarizer, query=query, content=cleaned_content)
            summary = pred.summary
            summaries.append(f"Source: {title} ({url})\nSummary: {summary}\n")
        except Exception as e:
            logger.error(f"Summarization failed for {url}: {e}")
            summaries.append(f"Source: {title} ({url})\nSummarization failed.\n")
            
    return "\n---\n".join(summaries)

from tenacity import AsyncRetrying, stop_after_attempt, wait_exponential, retry_if_exception_type

# ... imports ...

async def perform_web_search(query: str, max_results: int = 10) -> str:
    """Perform a deep web search: search -> scrape -> summarize.
    
    Args:
        query: The search query string.
        max_results: Maximum number of links to scrape (default 10).
        
    Returns:
        Aggregated summaries of the scraped content.
    """
    logger.info(f"Performing deep web search for: {query}")
    
    async for attempt in AsyncRetrying(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True
    ):
        with attempt:
            try:
                # Synchronous part: searching via generic DDGS (blocking?)
                # DDGS seems to be synchronous. We should run it in a thread.
                # Or check if DDGS has async? It doesn't seem so standardly.
                
                # Run search in thread
                logging.info("Running DDGS search...")
                def run_search():
                    with DDGS() as ddgs:
                         return list(ddgs.text(query, max_results=max_results))
                
                results = await asyncio.to_thread(run_search)
                
                if not results:
                    return "No results found."
                    
                # Async part: scraping
                aggregated_summary = await process_search_results(query, results)
                
                return aggregated_summary
                
            except Exception as e:
                logger.error(f"Search attempt failed: {e}")
                raise
