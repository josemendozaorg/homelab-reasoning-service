import asyncio
from src.reasoning.llm import llm

async def main():
    system_prompt = "You are a deep reasoning model. You explicitly separate your thinking process from your final answer using <think> tags."
    
    # Few-shot example construction
    prompt = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        "<|im_start|>user\nWhat is 2+2?<|im_end|>\n"
        "<|im_start|>assistant\n<think>\nThe user is asking for the sum of 2 and 2.\nThis is a basic arithmetic operation.\n2 plus 2 equals 4.\n</think>\n4<|im_end|>\n"
        "<|im_start|>user\nHow many r's are in the word strawberry? Think step by step.<|im_end|>\n"
        "<|im_start|>assistant\n<think>"
    )
    
    # NOTE: We are using generate_stream with the raw prompt to force completion
    print(f"Prompt with few-shot:\n{prompt}")
    
    print("\n--- Streaming Response ---")
    full_response = "<think>" # We prefilled this
    async for token in llm.generate_stream(prompt):
        print(token, end="", flush=True)
        full_response += token
    print("\n\n--- End Response ---")

    if "<think>" in full_response:
        print("\nSUCCESS: <think> tags detected.")
    else:
        print("\nWARNING: No <think> tags detected.")

if __name__ == "__main__":
    asyncio.run(main())
