import asyncio
import json
import httpx
import os
from datetime import datetime
from typing import Dict, Any

class LLMEvaluator:
    def __init__(self, ollama_url: str = "http://192.168.0.140:11434", model: str = "deepseek-r1:14b"):
        self.ollama_url = ollama_url
        self.model = model
        self.client = httpx.AsyncClient(timeout=60.0)

    def analyze_trace(self, trace: list, answer: str) -> Dict[str, Any]:
        """Analyze the reasoning trace for scaling metrics."""
        total_think_tokens = 0
        iterations = []
        
        for item in trace:
            if "[Iteration" in item:
                # Basic token counting approximation
                tokens = len(item.split())
                total_think_tokens += tokens
                iterations.append(tokens)
        
        answer_tokens = len(answer.split())
        
        return {
            "total_reasoning_words": total_think_tokens,
            "answer_words": answer_tokens,
            "reasoning_density": total_think_tokens / max(answer_tokens, 1),
            "iteration_distribution": iterations
        }

    async def score_answer(self, query: str, expected: str, actual: str, trace: list) -> Dict[str, Any]:
        """Use LLM to score the actual answer against the expected answer with deep reasoning analysis."""
        
        metrics = self.analyze_trace(trace, actual)
        
        prompt = f"""You are a SOTA reasoning evaluator. Compare the Actual Answer to the Expected Answer for the given Question. 
Evaluate the entire reasoning process if provided.

Question: {query}
Expected Answer: {expected}
Actual Answer: {actual}
Reasoning Trace Summary: {metrics['total_reasoning_words']} words of internal thought across {len(metrics['iteration_distribution'])} steps.

Criteria:
1. Accuracy (0-10): Factual correctness.
2. Depth (0-10): How deeply the model explored sub-problems and edge cases.
3. Feasibility (0-10): Real-world viability of the plan/answer.
4. Reasoning Quality (0-10): Is the thinking logical, or just "yapping"?

Output JSON format only:
{{
  "accuracy_score": (0-10),
  "depth_score": (0-10),
  "feasibility_score": (0-10),
  "reasoning_quality": (0-10),
  "is_correct": (true/false),
  "explanation": "concise explanation of strengths and weaknesses"
}}
"""
        try:
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "format": "json"
            }
            response = await self.client.post(f"{self.ollama_url}/api/chat", json=payload)
            response.raise_for_status()
            result = response.json()
            content = result.get("message", {}).get("content", "{}")
            eval_data = json.loads(content)
            eval_data.update(metrics) # Inline the metrics
            return eval_data
        except Exception as e:
            return {
                "accuracy_score": 0,
                "is_correct": False,
                "explanation": f"Evaluation failed: {str(e)}"
            }

    async def evaluate_results_file(self, results_path: str):
        with open(results_path, 'r') as f:
            data = json.load(f)
            
        evaluated_results = []
        correct_count = 0
        total_score = 0
        
        print(f"Evaluating results from {results_path}...")
        
        for res in data["results"]:
            if res.get("status") == "error":
                res["eval"] = {"accuracy_score": 0, "is_correct": False, "explanation": "Error in generation"}
                evaluated_results.append(res)
                continue
                
            eval_data = await self.score_answer(
                query=res["query"],
                expected=res["expected_answer"],
                actual=res["final_answer"],
                trace=res.get("reasoning_trace", [])
            )
            res["eval"] = eval_data
            evaluated_results.append(res)
            
            if eval_data.get("is_correct"):
                correct_count += 1
            total_score += eval_data.get("accuracy_score", 0)
            
        summary = {
            "total_items": len(data["results"]),
            "correct_items": correct_count,
            "accuracy": (correct_count / len(data["results"])) * 100 if data["results"] else 0,
            "average_score": (total_score / len(data["results"])) if data["results"] else 0,
            "average_latency": sum(r["duration"] for r in data["results"]) / len(data["results"]) if data["results"] else 0,
            "average_iterations": sum(r.get("iterations", 0) for r in data["results"]) / len(data["results"]) if data["results"] else 0,
            "average_reasoning_density": sum(r["eval"].get("reasoning_density", 0) for r in evaluated_results) / len(data["results"]) if data["results"] else 0
        }
        
        data["evaluated_at"] = datetime.now().isoformat()
        data["summary"] = summary
        data["results"] = evaluated_results
        
        eval_path = results_path.replace(".json", "_evaluated.json")
        with open(eval_path, 'w') as f:
            json.dump(data, f, indent=2)
            
        print(f"Evaluation complete. Summary: {summary}")
        print(f"Evaluated report saved to {eval_path}")
        return summary

if __name__ == "__main__":
    import sys
    results_file = sys.argv[1]
    evaluator = LLMEvaluator()
    asyncio.run(evaluator.evaluate_results_file(results_file))
