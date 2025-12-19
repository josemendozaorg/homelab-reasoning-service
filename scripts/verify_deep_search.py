import asyncio
import logging
import dspy
from src.config import settings
from src.reasoning.tools import perform_web_search

# Configure logging
logging.basicConfig(level=logging.INFO)

# Configure DSPy
lm = dspy.LM(
    f"ollama/{settings.ollama_model}",
    api_base=settings.ollama_base_url,
    temperature=settings.temperature,
    max_tokens=settings.max_context_tokens,
    api_key="nomatter"
)
dspy.configure(lm=lm)

def test_deep_search():
    query = "DeepSeek-R1 properties"
    print(f"Testing deep search for: {query}")
    
    # This calls the synchronous wrapper which runs asyncio.run internally
    result = perform_web_search(query, max_results=3) # Limit to 3 for speed
    
    print("\n\n=== Search Results Summary ===\n")
    print(result)
    
    if "Summary:" in result:
        print("\n✅ Verification PASSED: Summaries generated.")
    else:
        print("\n❌ Verification FAILED: No summaries found.")

if __name__ == "__main__":
    test_deep_search()
