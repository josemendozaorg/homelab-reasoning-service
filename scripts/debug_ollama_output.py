import asyncio
import httpx
import json

async def test_raw_ollama():
    url = "http://192.168.0.140:11434/api/generate"
    payload = {
        "model": "deepseek-r1:14b",
        "prompt": "How many r's are in Strawberry?",
        "stream": False,
        "raw": True
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=payload, timeout=60.0)
        print(f"Status: {response.status_code}")
        content = response.json().get("response", "")
        print(f"Content Length: {len(content)}")
        print("--- CONTENT START ---")
        print(content)
        print("--- CONTENT END ---")
        
        # Check for think tags
        if "<think>" in content:
            print("Found <think> tag!")
        else:
            print("NO <think> tag found.")

if __name__ == "__main__":
    asyncio.run(test_raw_ollama())
