import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import dspy
from src.config import settings

# Initialize real LM
lm = dspy.LM(
    f"ollama/{settings.ollama_model}",
    api_base=settings.ollama_base_url,
    temperature=settings.temperature,
    max_tokens=2000,
    api_key="nomatter"
)
dspy.configure(lm=lm)

def test_direct_call():
    print(f"Testing direct call to {settings.ollama_model}...")
    
    messages = [
        {"role": "system", "content": """You are a helpful assistant.
Instructions:
1. Use <think> tags for internal reasoning.
2. TRUST THE Context. It is the absolute source of truth.
3. Provide the answer text ONLY. Start with 'Final Answer:'."""},
        {"role": "user", "content": "Context: Current Date: 2025-01-01\nQuestion: What is the date?"}
    ]
    
    try:
        # Direct call bypassing Signatures/Adapters
        response = lm(messages=messages)
        print("\n=== RAW RESPONSE START ===")
        print(response)
        print("=== RAW RESPONSE END ===")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_direct_call()
