"""Tools available to the reasoning agent."""
import logging
from ddgs import DDGS
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger(__name__)

def perform_web_search(query: str, max_results: int = 3) -> str:
    """Perform a web search using DuckDuckGo.
    
    Args:
        query: The search query string.
        max_results: Maximum number of results to return.
        
    Returns:
        Formatted string of search results.
    """
    logger.info(f"Performing web search for: {query}")
    return _perform_web_search_with_retry(query, max_results)

@retry(
    stop=stop_after_attempt(3), 
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(Exception),
    reraise=True
)
def _perform_web_search_with_retry(query: str, max_results: int) -> str:
    """Internal search function with retry logic."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
            
        if not results:
            return "No results found."
            
        formatted_results = []
        for i, res in enumerate(results, 1):
            title = res.get('title', 'No Title')
            body = res.get('body', 'No Description')
            href = res.get('href', '#')
            formatted_results.append(f"Result {i}:\nTitle: {title}\nURL: {href}\nSnippet: {body}\n")
            
        return "\n".join(formatted_results)
        
        return "\n".join(formatted_results)
        
    except Exception as e:
        logger.error(f"Search attempt failed: {e}")
        raise # Reraise to trigger retry
