import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import dspy
from src.config import settings
from src.reasoning.signatures import ReasonSignature

# Mock LM that just captures prompts
class InspectLM(dspy.LM):
    def __init__(self):
        super().__init__("inspect-mock")
        self.history = []

    def __call__(self, prompt=None, messages=None, **kwargs):
        if messages:
            self.history.append(str(messages))
        if prompt:
            self.history.append(prompt)
        return [{"text": "<think>Mock reasoning</think>Final Answer: Mock Answer"}]

    def basic_request(self, prompt, **kwargs):
        return self(prompt, **kwargs)

def inspect_reason_prompt():
    # Setup mock LM
    mock_lm = InspectLM()
    dspy.configure(lm=mock_lm)
    
    # Instantiate predictor
    predictor = dspy.Predict(ReasonSignature)
    
    # Run
    print("Running predictor...")
    try:
        predictor(
            question="What is the date?",
            context="Current Date: 2025-01-01"
        )
    except Exception as e:
        print(f"Caught expected error: {e}")
    
    # Print the raw prompt
    print("\n=== GENERATED PROMPT START ===")
    print(mock_lm.history[-1])
    print("=== GENERATED PROMPT END ===")

if __name__ == "__main__":
    inspect_reason_prompt()
