import asyncio
import sys
import os
from evals.framework.runner import EvalRunner
from evals.framework.evaluator import LLMEvaluator

async def main():
    dataset = sys.argv[1] if len(sys.argv) > 1 else "evals/datasets/reasoning_basics.json"
    
    # 1. Run the benchmark
    runner = EvalRunner()
    results, results_path = await runner.run_dataset(dataset)
    
    # 2. Evaluate the results
    evaluator = LLMEvaluator()
    summary = await evaluator.evaluate_results_file(results_path)
    
    # 3. Print final report
    print("\n" + "="*50)
    print("EVALUATION REPORT")
    print("="*50)
    print(f"Dataset:    {dataset}")
    print(f"Total:      {summary['total_items']}")
    print(f"Correct:    {summary['correct_items']} ({summary['accuracy']:.1f}%)")
    print(f"Avg Score:  {summary['average_score']:.2f}/10")
    print(f"Avg Time:   {summary['average_latency']:.2f}s")
    print(f"Avg Iters:  {summary['average_iterations']:.2f}")
    print("="*50)
    print(f"Detailed report: {results_path.replace('.json', '_evaluated.json')}")

if __name__ == "__main__":
    asyncio.run(main())
