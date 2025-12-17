import dspy
import inspect

print("DSPy version:", dspy.__version__)
print("DSPy attributes:", dir(dspy))

# Check for Ollama-related classes
for attr in dir(dspy):
    if "ollama" in attr.lower():
        print(f"Found candidate: {attr}")

# Try to find where LM classes are
if hasattr(dspy, "LM"):
    print("dspy.LM exists")

# Check dspy.aio or similar
