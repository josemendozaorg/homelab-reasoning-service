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

### Local Development

```bash
# Clone repository
git clone https://github.com/josemendozaorg/homelab-reasoning-service.git
cd homelab-reasoning-service

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your Ollama endpoint

# Run service
python -m src.main
```

### Docker

```bash
# Build and run
docker compose up --build

# Or just build
docker build -t reasoning-service .
docker run -p 8080:8080 \
  -e REASONING_OLLAMA_BASE_URL=http://192.168.0.140:11434 \
  reasoning-service
```

### Coolify Deployment

1. Access Coolify UI: http://192.168.0.160:8000
2. Create new application from GitHub
3. Select repository: `josemendozaorg/homelab-reasoning-service`
4. Build pack: Dockerfile
5. Configure environment variables:
   ```
   REASONING_OLLAMA_BASE_URL=http://192.168.0.140:11434
   REASONING_OLLAMA_MODEL=deepseek-r1:14b
   REASONING_MAX_REASONING_ITERATIONS=5
   ```
6. Deploy

## API Usage

### Submit Reasoning Task

```bash
curl -X POST http://localhost:8080/v1/reason \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the implications of Godel'\''s incompleteness theorems for AGI?",
    "max_iterations": 5
  }'
```

### Response

```json
{
  "query": "What are the implications...",
  "reasoning_trace": [
    "[Iteration 1]\nFirst, let me recall Godel's theorems...",
    "[Iteration 2]\nConsidering implications for AGI..."
  ],
  "final_answer": "Godel's incompleteness theorems suggest...",
  "iterations": 3,
  "is_approved": true
}
```

### Health Check

```bash
curl http://localhost:8080/health
```

```json
{
  "status": "healthy",
  "model": "deepseek-r1:14b",
  "ollama_connected": true
}
```

## Configuration

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
