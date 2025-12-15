# LangGraph Reasoning Service

Inference-Time Scaling with Self-Correcting Reasoning for homelab deployment.

## Overview

This service implements **System 2 reasoning** using:
- **DeepSeek-R1:14B** - Native `<think>` token support for chain-of-thought
- **LangGraph** - Stateful workflow management with cyclic graphs
- **Self-Correction Loop** - Reason → Critique → Refine → Repeat

Designed for complex reasoning tasks that benefit from deliberative, iterative problem-solving.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│           LangGraph Reasoning Service               │
│  ┌─────────────────────────────────────────────┐   │
│  │  ENTRY → REASON → CRITIQUE → DECIDE         │   │
│  │                      ↑         │             │   │
│  │                      └─────────┴─→ END      │   │
│  └─────────────────────────────────────────────┘   │
└───────────────────────┬─────────────────────────────┘
                        │ HTTP
                        ▼
┌─────────────────────────────────────────────────────┐
│              Ollama (192.168.0.140:11434)           │
│              DeepSeek-R1:14B                        │
└─────────────────────────────────────────────────────┘
```

## Quick Start


| Variable | Default | Description |
|----------|---------|-------------|
| `REASONING_OLLAMA_BASE_URL` | `http://192.168.0.140:11434` | Ollama API endpoint |
| `REASONING_OLLAMA_MODEL` | `deepseek-r1:14b` | Model to use for reasoning |
| `REASONING_MAX_REASONING_ITERATIONS` | `5` | Maximum self-correction iterations |
| `REASONING_MAX_CONTEXT_TOKENS` | `16000` | Maximum context window |
| `REASONING_TEMPERATURE` | `0.7` | Sampling temperature |
| `REASONING_API_HOST` | `0.0.0.0` | API bind host |
| `REASONING_API_PORT` | `8080` | API port |

## How It Works

### The Self-Correction Loop

1. **REASON**: The model generates an answer using `<think>` tokens for chain-of-thought reasoning
2. **CRITIQUE**: A critical evaluation identifies logical errors, gaps, or improvements
3. **DECIDE**: If critique says "APPROVED" or max iterations reached, finalize; otherwise continue
4. **REFINE**: The model improves the answer based on critique feedback
5. **REPEAT**: Back to CRITIQUE until approved

### Why DeepSeek-R1?

DeepSeek-R1 is trained to emit `<think>...</think>` tokens naturally, making it ideal for:
- Explicit reasoning traces
- Self-questioning during generation
- Error catching before final output

Q4_K_M quantization fits in ~10GB VRAM on a 16GB GPU, leaving room for context.

## Project Structure

```
homelab-reasoning-service/
├── src/
│   ├── main.py           # FastAPI application
│   ├── config.py         # Settings management
│   ├── api/
│   │   ├── routes.py     # API endpoints
│   │   └── models.py     # Request/Response schemas
│   ├── reasoning/
│   │   ├── graph.py      # LangGraph workflow
│   │   ├── nodes.py      # Node implementations
│   │   ├── state.py      # State schema
│   │   └── prompts.py    # System prompts
│   └── llm/
│       └── ollama_client.py  # Ollama API wrapper
├── tests/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Future Enhancements

- [ ] MCTS (Monte Carlo Tree Search) for complex problem solving
- [ ] Tree of Thoughts parallel exploration
- [ ] Docker sandbox for code execution
- [ ] Process Reward Model integration
- [ ] Streaming responses
