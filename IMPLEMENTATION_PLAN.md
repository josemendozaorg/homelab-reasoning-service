# Implementation Plan: Ollama Model Selection in UI

## Overview

Enable users to select from available Ollama models directly in the web UI. This requires changes to both backend (FastAPI/LangGraph) and frontend (vanilla JS).

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Frontend (UI)                           │
│  ┌─────────────────────┐    ┌─────────────────────────────────┐ │
│  │   Model Dropdown    │───▶│  localStorage (persist choice)  │ │
│  └─────────────────────┘    └─────────────────────────────────┘ │
│           │                                                      │
│           │ 1. GET /v1/models (on load)                         │
│           │ 2. Include model in POST /v1/reason/stream          │
│           ▼                                                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ HTTP
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Backend (FastAPI)                          │
│  ┌─────────────────────┐    ┌─────────────────────────────────┐ │
│  │  GET /v1/models     │───▶│  OllamaClient.list_models()     │ │
│  └─────────────────────┘    └─────────────────────────────────┘ │
│  ┌─────────────────────┐    ┌─────────────────────────────────┐ │
│  │  ReasoningRequest   │───▶│  ReasoningState (with model)    │ │
│  │    + model field    │    └─────────────────────────────────┘ │
│  └─────────────────────┘                   │                    │
│                                            ▼                    │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │            LangGraph (uses config["configurable"]["model"]) ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ HTTP (Ollama API)
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Ollama Service                               │
│  GET /api/tags → list available models                          │
│  POST /api/chat → inference with specified model                │
└─────────────────────────────────────────────────────────────────┘
```

---

## Step 1: Add `list_models()` to OllamaClient

**File:** `src/llm/ollama_client.py`

Add a new method to fetch available models from Ollama:

```python
async def list_models(self) -> list[dict]:
    """Fetch list of available models from Ollama.

    Returns:
        List of model info dicts with 'name', 'size', 'modified_at' etc.
    """
    try:
        async with httpx.AsyncClient(base_url=self.base_url, timeout=10.0) as client:
            response = await client.get("/api/tags")
            response.raise_for_status()
            data = response.json()
            return data.get("models", [])
    except Exception as e:
        logger.warning(f"Failed to list models: {e}")
        return []
```

---

## Step 2: Add API Models for Models Endpoint

**File:** `src/api/models.py`

Add new response model:

```python
class ModelInfo(BaseModel):
    """Information about an available model."""
    name: str = Field(description="Model name (e.g., 'deepseek-r1:14b')")
    size: int = Field(description="Model size in bytes")
    modified_at: str = Field(description="Last modified timestamp")

class ModelsResponse(BaseModel):
    """Response model for listing available models."""
    models: list[ModelInfo] = Field(description="List of available models")
    default: str = Field(description="Default model from configuration")
```

Update `ReasoningRequest` to accept optional model:

```python
class ReasoningRequest(BaseModel):
    query: str = Field(...)
    model: Optional[str] = Field(
        default=None,
        description="Model to use (defaults to server configuration)"
    )
    max_iterations: Optional[int] = Field(...)
    temperature: Optional[float] = Field(...)
    history: list[dict] = Field(default=[])
```

---

## Step 3: Add GET /v1/models Endpoint

**File:** `src/api/routes.py`

```python
@router.get("/v1/models", response_model=ModelsResponse)
async def list_models() -> ModelsResponse:
    """List available Ollama models.

    Returns:
        List of models with their metadata and the default model.
    """
    client = OllamaClient()
    models = await client.list_models()

    return ModelsResponse(
        models=[
            ModelInfo(
                name=m["name"],
                size=m.get("size", 0),
                modified_at=m.get("modified_at", "")
            )
            for m in models
        ],
        default=settings.ollama_model
    )
```

---

## Step 4: Update ReasoningState to Include Model

**File:** `src/reasoning/state.py`

Add model field to state:

```python
class ReasoningState(TypedDict):
    # ... existing fields ...
    model: Optional[str]  # Model to use for this reasoning task
```

Update `create_initial_state`:

```python
def create_initial_state(query: str, history: list[dict] = [], model: str = None) -> ReasoningState:
    return ReasoningState(
        # ... existing fields ...
        model=model  # Will use default if None
    )
```

---

## Step 5: Update Routes to Pass Model

**File:** `src/api/routes.py`

Update the streaming endpoint to pass model through LangGraph config:

```python
@router.post("/v1/reason/stream")
async def reason_stream(request: ReasoningRequest, req: Request):
    async def event_generator():
        # Use requested model or fall back to default
        model = request.model or settings.ollama_model

        initial_state = create_initial_state(
            request.query,
            request.history,
            model=model
        )

        # Pass model via LangGraph config for node access
        config = {
            "recursion_limit": 150,
            "configurable": {"model": model}
        }

        graph = get_reasoning_graph()
        async for event in graph.astream_events(initial_state, version="v2", config=config):
            # ... existing event handling ...
```

---

## Step 6: Update Nodes to Use Model from Config

**File:** `src/reasoning/nodes.py` (or wherever nodes create OllamaClient)

Update node functions to extract model from config:

```python
async def reason_node(state: ReasoningState, config: RunnableConfig):
    # Get model from config (passed from route)
    model = config.get("configurable", {}).get("model") or settings.ollama_model

    async with OllamaClient(model=model) as client:
        # ... use client ...
```

---

## Step 7: Add Model Selector to UI HTML

**File:** `src/static/index.html`

Replace the static model info with a dropdown:

```html
<div class="input-options">
    <label class="toggle-switch">
        <input type="checkbox" id="showTrace" checked>
        <span class="toggle-label">Show Reasoning Trace</span>
    </label>

    <!-- New Model Selector -->
    <div class="model-selector">
        <label for="modelSelect">Model:</label>
        <select id="modelSelect" class="model-dropdown">
            <option value="">Loading...</option>
        </select>
    </div>
</div>
```

---

## Step 8: Add Model Selector Styles

**File:** `src/static/style.css`

Add styles for the model dropdown:

```css
/* Model Selector */
.model-selector {
    display: flex;
    align-items: center;
    gap: var(--spacing-sm);
}

.model-selector label {
    color: var(--nb-text-secondary);
    font-size: 0.875rem;
}

.model-dropdown {
    background: var(--nb-bg-tertiary);
    color: var(--nb-text-primary);
    border: 1px solid var(--nb-border-subtle);
    border-radius: var(--radius-sm);
    padding: var(--spacing-xs) var(--spacing-sm);
    font-family: var(--font-mono);
    font-size: 0.75rem;
    cursor: pointer;
    min-width: 180px;
    transition: border-color var(--transition-fast);
}

.model-dropdown:hover {
    border-color: var(--nb-border-active);
}

.model-dropdown:focus {
    outline: none;
    border-color: var(--nb-accent-primary);
}

.model-dropdown option {
    background: var(--nb-bg-secondary);
    color: var(--nb-text-primary);
}
```

---

## Step 9: Implement Model Selection Logic in JS

**File:** `src/static/app.js`

Add model fetching and selection:

```javascript
const modelSelect = document.getElementById('modelSelect');
let selectedModel = localStorage.getItem('selectedModel') || null;

// Fetch available models on page load
async function fetchModels() {
    try {
        const res = await fetchWithRetry('/v1/models');
        const data = await res.json();

        modelSelect.innerHTML = '';

        data.models.forEach(model => {
            const option = document.createElement('option');
            option.value = model.name;
            option.textContent = model.name;

            // Mark default model
            if (model.name === data.default) {
                option.textContent += ' (default)';
            }

            modelSelect.appendChild(option);
        });

        // Restore previous selection or use default
        if (selectedModel && data.models.some(m => m.name === selectedModel)) {
            modelSelect.value = selectedModel;
        } else {
            modelSelect.value = data.default;
            selectedModel = data.default;
        }

    } catch (e) {
        console.error("Failed to fetch models:", e);
        modelSelect.innerHTML = '<option value="">Error loading models</option>';
    }
}

// Save selection on change
modelSelect.addEventListener('change', () => {
    selectedModel = modelSelect.value;
    localStorage.setItem('selectedModel', selectedModel);
});

// Call on page load
fetchModels();
```

Update the submit handler to include model:

```javascript
// In the submit handler, update the fetch body:
body: JSON.stringify({
    query: query,
    model: selectedModel,  // Add selected model
    max_iterations: 5,
    history: chatHistory
})
```

---

## Step 10: Update /api/info to Include Model List

**File:** `src/main.py`

Optionally enhance the info endpoint:

```python
@app.get("/api/info")
async def info():
    return {
        "version": "1.0.0",
        "commit": settings.commit_hash,
        "default_model": settings.ollama_model
    }
```

---

## Summary of File Changes

| File | Changes |
|------|---------|
| `src/llm/ollama_client.py` | Add `list_models()` method |
| `src/api/models.py` | Add `ModelInfo`, `ModelsResponse`; update `ReasoningRequest` with `model` field |
| `src/api/routes.py` | Add `GET /v1/models` endpoint; pass model via config to graph |
| `src/reasoning/state.py` | Add `model` field to `ReasoningState`; update `create_initial_state` |
| `src/reasoning/nodes.py` | Extract model from config in node functions |
| `src/static/index.html` | Replace static model info with dropdown |
| `src/static/style.css` | Add model selector styles |
| `src/static/app.js` | Add `fetchModels()`, model selection logic, persist to localStorage |

---

## Testing Checklist

- [ ] `GET /v1/models` returns list of available models
- [ ] Default model is correctly identified in response
- [ ] Model dropdown populates on page load
- [ ] Selected model persists across page refreshes (localStorage)
- [ ] Reasoning requests use the selected model
- [ ] Fallback to default model when none selected
- [ ] Error handling when Ollama is unreachable

---

## Notes

1. **Model compatibility:** Not all models support the `<think>` tag format used by DeepSeek-R1. Consider adding a compatibility indicator or warning for non-reasoning models.

2. **Model refresh:** Consider adding a refresh button to reload available models without page refresh.

3. **Model size display:** The dropdown could show model sizes in a human-readable format (GB) to help users choose.
