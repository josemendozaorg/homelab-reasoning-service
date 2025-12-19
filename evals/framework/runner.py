import asyncio
import json
import time
import httpx
import os
from datetime import datetime

class EvalRunner:
    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=300.0) # High timeout for reasoning

    async def run_query(self, query: str, max_iterations: int = 5):
        url = f"{self.base_url}/v1/reason"
        start_time = time.time()
        
        try:
            response = await self.client.post(
                url, 
                json={"query": query, "max_iterations": max_iterations}
            )
            response.raise_for_status()
            data = response.json()
            duration = time.time() - start_time
            
            return {
                "status": "success",
                "duration": duration,
                "iterations": data.get("iterations", 0),
                "final_answer": data.get("final_answer", ""),
                "is_approved": data.get("is_approved", False),
                "reasoning_trace": data.get("reasoning_trace", [])
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "duration": time.time() - start_time
            }

    async def run_dataset(self, dataset_path: str, output_dir: str = "evals/results"):
        with open(dataset_path, 'r') as f:
            dataset = json.load(f)
            
        results = []
        print(f"Running evaluation on {len(dataset)} items...")
        
        for item in dataset:
            print(f"Evaluating: {item['id']} - {item['query'][:50]}...")
            res = await self.run_query(item['query'])
            res["id"] = item["id"]
            res["query"] = item["query"]
            res["expected_answer"] = item["expected_answer"]
            results.append(res)
            
        # Save results
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"eval_{os.path.basename(dataset_path).split('.')[0]}_{timestamp}.json"
        output_path = os.path.join(output_dir, filename)
        
        with open(output_path, 'w') as f:
            json.dump({
                "timestamp": timestamp,
                "dataset": dataset_path,
                "results": results
            }, f, indent=2)
            
        print(f"Results saved to {output_path}")
        return results, output_path

if __name__ == "__main__":
    import sys
    dataset_file = sys.argv[1] if len(sys.argv) > 1 else "evals/datasets/reasoning_basics.json"
    runner = EvalRunner()
    asyncio.run(runner.run_dataset(dataset_file))
