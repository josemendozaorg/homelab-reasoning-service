import httpx
import json
import asyncio

async def test_stream():
    url = "http://localhost:8080/v1/reason/stream"
    payload = {
        "query": "Hello",
        "max_iterations": 1,
        "history": []
    }
    
    print(f"Connecting to {url}...")
    async with httpx.AsyncClient(timeout=30.0) as client:
        async with client.stream("POST", url, json=payload) as response:
            print(f"Status: {response.status_code}")
            async for line in response.aiter_lines():
                print(f"RAW LINE: {line!r}")
                if line.startswith("data: "):
                    data_str = line[6:]
                    try:
                        parsed = json.loads(data_str)
                        print(f"PARSED OK: {parsed.keys()}")
                    except json.JSONDecodeError as e:
                        print(f"PARSE ERROR: {e}")

if __name__ == "__main__":
    asyncio.run(test_stream())
