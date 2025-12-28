# Search Implementation Improvement Plan

## Executive Summary

The current search implementation is **slow and wasteful**. It scrapes 10 URLs and makes 10 LLM summarization calls per search, when DuckDuckGo already provides useful snippets. Based on research into Perplexity, Agentic RAG, and LangChain best practices, this plan proposes a tiered approach that uses snippets first and scrapes selectively.

---

## Current Issues

| Issue | Impact | Evidence |
|-------|--------|----------|
| **Scrapes ALL 10 URLs** | 30-60 sec latency | `max_results: int = 10` in `perform_web_search` |
| **LLM summarization for EVERY page** | 10+ LLM calls/search | Loop in `process_search_results` |
| **Ignores DuckDuckGo snippets** | Wastes existing data | `results[i].get('body')` never used |
| **No relevance filtering** | Scrapes irrelevant pages | No scoring before scraping |
| **No caching** | Re-searches same queries | No cache implementation |
| **Blocking DDGS call** | Event loop blocked | Uses `asyncio.to_thread` but still slow |

### Current Flow (Slow)
```
Query → DuckDuckGo (10 results) → Scrape ALL 10 URLs → Summarize ALL 10 with LLM → Return
         ~2 sec                     ~10-30 sec             ~20-40 sec
```

---

## Research Findings

### From Perplexity Architecture
> "The system does not pass the full text of the retrieved web pages to the generative model. The indexing and retrieval infrastructure divides documents up into fine-grained units."

Source: [Perplexity Search API Architecture](https://research.perplexity.ai/articles/architecting-and-evaluating-an-ai-first-search-api)

### From Agentic RAG Survey
> "Self-RAG trains models to decide when to retrieve, and to critique their own outputs—boosting factuality and citation accuracy."

Source: [Agentic RAG Survey](https://arxiv.org/abs/2501.09136)

### From LangChain Best Practices
> "Combining a small amount of manual scraping while letting GPT handle the details is a good combination of convenient and cost-effective."

Source: [LangChain Web Scraping](https://python.langchain.com/docs/use_cases/web_scraping/)

---

## Proposed Architecture

### New Flow (Fast)
```
Query → DuckDuckGo → Return Snippets → LLM Assesses Sufficiency
                          ↓                      ↓
                    [If Sufficient]        [If Insufficient]
                          ↓                      ↓
                    Return Snippets     Scrape Top 3 Relevant URLs
                                                 ↓
                                        Return Combined
```

### Tiered Retrieval Strategy

| Tier | When Used | URLs Scraped | LLM Calls | Latency |
|------|-----------|--------------|-----------|---------|
| **Tier 1: Snippets Only** | Snippets answer query | 0 | 0 | ~2 sec |
| **Tier 2: Selective Scrape** | Need more detail | 2-3 | 0-1 | ~5-10 sec |
| **Tier 3: Deep Scrape** | Complex research | 5-7 | 0 | ~15-20 sec |

---

## Implementation Plan

### Phase 1: Use Snippets First (HIGH IMPACT, LOW EFFORT)

**Change:** DuckDuckGo returns `body` snippets. Use them!

```python
# Current (ignores snippets)
for res in results:
    url = res.get('href')
    # scrapes URL...

# Proposed
async def perform_web_search(query: str, config: RunnableConfig = None) -> str:
    results = await search_ddg(query, max_results=5)  # Reduced from 10

    # Build snippet-based response FIRST
    snippets = []
    for res in results:
        snippets.append({
            "title": res.get("title", ""),
            "url": res.get("href", ""),
            "snippet": res.get("body", ""),  # USE THE SNIPPET!
        })

    # Format for LLM
    formatted = format_search_results(snippets)

    # Return snippets - let the reasoning LLM decide if more detail needed
    return formatted
```

**Expected improvement:** 10x faster for most queries (2 sec vs 30 sec)

---

### Phase 2: Selective Scraping (MEDIUM EFFORT)

**Change:** Only scrape when snippets are insufficient.

```python
async def perform_web_search(
    query: str,
    depth: str = "snippets",  # "snippets" | "selective" | "deep"
    config: RunnableConfig = None
) -> str:
    results = await search_ddg(query, max_results=5)

    # Always return snippets
    snippet_response = format_snippets(results)

    if depth == "snippets":
        return snippet_response

    if depth == "selective":
        # Score relevance and scrape top 2-3
        scored = score_relevance(results, query)
        top_urls = [r["href"] for r in scored[:3]]
        scraped = await scrape_urls(top_urls)
        return snippet_response + "\n\n" + scraped

    if depth == "deep":
        # Scrape more for complex research
        scraped = await scrape_urls([r["href"] for r in results[:7]])
        return snippet_response + "\n\n" + scraped
```

**Integration with MCTS:** The expand node can request different depths based on query complexity.

---

### Phase 3: Remove Per-Page Summarization (HIGH IMPACT)

**Change:** Let the reasoning LLM synthesize, don't pre-summarize.

**Current:** 10 LLM calls to summarize 10 pages
**Proposed:** 0 LLM calls, pass clean text directly

```python
# Remove this entire loop that calls LLM for each page:
# for i, content in enumerate(pages_content):
#     ...
#     summary = await llm.chat(...)  # REMOVE
#     ...

# Instead, just clean and concatenate
def process_scraped_content(pages: list, max_chars: int = 8000) -> str:
    """Clean and combine scraped content without LLM summarization."""
    combined = []
    chars_used = 0

    for page in pages:
        if chars_used >= max_chars:
            break
        cleaned = clean_text(page["html"])
        # Take first 2000 chars per page
        chunk = cleaned[:2000]
        combined.append(f"Source: {page['title']}\n{chunk}")
        chars_used += len(chunk)

    return "\n\n---\n\n".join(combined)
```

**Expected improvement:** Eliminates 10+ LLM calls per search

---

### Phase 4: Add Simple Caching (MEDIUM EFFORT)

```python
from functools import lru_cache
from datetime import datetime, timedelta
import hashlib

# Simple in-memory cache with TTL
_search_cache = {}
_cache_ttl = timedelta(minutes=10)

async def perform_web_search_cached(query: str, **kwargs) -> str:
    cache_key = hashlib.md5(query.encode()).hexdigest()

    if cache_key in _search_cache:
        result, timestamp = _search_cache[cache_key]
        if datetime.now() - timestamp < _cache_ttl:
            logger.info(f"Cache hit for query: {query[:30]}...")
            return result

    # Cache miss - perform search
    result = await perform_web_search(query, **kwargs)
    _search_cache[cache_key] = (result, datetime.now())

    return result
```

---

### Phase 5: Source Quality Scoring (LOWER PRIORITY)

```python
# Prefer authoritative domains
TRUSTED_DOMAINS = {
    "wikipedia.org": 1.0,
    "gov": 0.9,
    "edu": 0.9,
    "reuters.com": 0.85,
    "bbc.com": 0.85,
    # ... etc
}

def score_source_quality(url: str) -> float:
    """Score source reliability."""
    for domain, score in TRUSTED_DOMAINS.items():
        if domain in url:
            return score
    return 0.5  # Unknown source
```

---

## Comparison: Before vs After

| Metric | Current | Proposed (Tier 1) | Proposed (Tier 2) |
|--------|---------|-------------------|-------------------|
| URLs scraped | 10 | 0 | 2-3 |
| LLM calls per search | 10+ | 0 | 0 |
| Latency | 30-60 sec | ~2 sec | ~5-10 sec |
| Context quality | Summarized | Raw snippets | Snippets + detail |

---

## Integration with MCTS/LATS

The search depth should be controlled by the reasoning agent:

```python
# In mcts_expand_node or reason_node:
async def determine_search_depth(query: str, iteration: int) -> str:
    """Decide how deep to search based on context."""
    if iteration == 0:
        return "snippets"  # Start fast
    if "detailed" in query.lower() or "comprehensive" in query.lower():
        return "deep"
    return "selective"
```

---

## Migration Path

1. **Week 1:** Implement Phase 1 (snippets first) - Immediate 10x speedup
2. **Week 2:** Implement Phase 3 (remove summarization) - Major simplification
3. **Week 3:** Implement Phase 2 (selective scraping) - Smart depth control
4. **Week 4:** Implement Phase 4 (caching) - Efficiency for repeated queries
5. **Future:** Phase 5 (source quality) - Polish

---

## References

- [Perplexity Search API Architecture](https://research.perplexity.ai/articles/architecting-and-evaluating-an-ai-first-search-api)
- [Agentic RAG Survey (2025)](https://arxiv.org/abs/2501.09136)
- [LangChain Web Scraping Best Practices](https://python.langchain.com/docs/use_cases/web_scraping/)
- [ByteByteGo: How Perplexity Built an AI Google](https://blog.bytebytego.com/p/how-perplexity-built-an-ai-google)
- [Self-RAG: Learning to Retrieve, Generate, and Critique](https://arxiv.org/abs/2310.11511)
