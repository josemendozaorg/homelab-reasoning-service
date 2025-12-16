"""Tools available to the reasoning agent."""
import logging
from ddgs import DDGS

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
        
    except Exception as e:
        logger.error(f"Search failed: {e}")
        return f"Search failed: {str(e)}"
