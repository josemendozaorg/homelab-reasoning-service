const queryInput = document.getElementById('queryInput');
const submitBtn = document.getElementById('submitBtn');
const queryForm = document.getElementById('queryForm');
const chatContainer = document.getElementById('chatContainer');
const placeholder = document.getElementById('placeholder');
const showTraceCheckbox = document.getElementById('showTrace');
const appVersion = document.getElementById('appVersion');
const modelSelect = document.getElementById('modelSelect');
const fastModelSelect = document.getElementById('fastModelSelect');

// Settings Elements
const settingsBtn = document.getElementById('settingsBtn');
const quickSettingsBtn = document.getElementById('quickSettingsBtn'); // New button inside pill
const settingsModal = document.getElementById('settingsModal');
const closeSettings = document.getElementById('closeSettings');
const saveSettingsBtn = document.getElementById('saveSettings');
const searchProviderSelect = document.getElementById('searchProvider');
const searchProviderPill = document.getElementById('searchProviderPill'); // New pill dropdown

const tavilyConfig = document.getElementById('tavilyConfig');
const exaConfig = document.getElementById('exaConfig'); // New
const braveConfig = document.getElementById('braveConfig');
const googleConfig = document.getElementById('googleConfig');

const tavilyKeyInput = document.getElementById('tavilyKey');
const exaKeyInput = document.getElementById('exaKey'); // New
const braveKeyInput = document.getElementById('braveKey');
const googleKeyInput = document.getElementById('googleKey');
const googleCseIdInput = document.getElementById('googleCseId');


let chatHistory = []; // Local history state
let selectedModel = localStorage.getItem('selectedModel') || null;
let selectedFastModel = localStorage.getItem('selectedFastModel') || null;

// Initialize
async function fetchWithRetry(url, options = {}, retries = 3, backoff = 1000) {
    try {
        const response = await fetch(url, options);
        // Retry on transient server errors (502, 503, 504)
        if (!response.ok && (response.status === 502 || response.status === 503 || response.status === 504)) {
            throw new Error(`Server Error: ${response.status}`);
        }
        return response;
    } catch (error) {
        if (retries > 0) {
            console.warn(`Fetch failed, retrying... (${retries} attempts left)`, error);
            await new Promise(resolve => setTimeout(resolve, backoff));
            return fetchWithRetry(url, options, retries - 1, backoff * 2);
        }
        throw error;
    }
}

// Fetch and display version info
(async () => {
    try {
        const res = await fetchWithRetry('/api/info');
        const data = await res.json();
        if (data.version) {
            const commit = data.commit || 'dev';
            appVersion.textContent = `v${data.version} (${commit})`;
        }
    } catch (e) {
        console.error("Failed to fetch version:", e);
    }
})();

// Fetch available models and populate dropdown
async function fetchModels() {
    try {
        const res = await fetchWithRetry('/v1/models');
        const data = await res.json();

        modelSelect.innerHTML = '';
        fastModelSelect.innerHTML = '';

        if (data.models && data.models.length > 0) {
            // Populate Smart Model Dropdown
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

            // Populate Fast Model Dropdown
            data.models.forEach(model => {
                const option = document.createElement('option');
                option.value = model.name;
                option.textContent = model.name;

                // Mark default fast model
                if (model.name === data.default_fast) {
                    option.textContent += ' (default)';
                }

                fastModelSelect.appendChild(option);
            });


            // Restore previous selection or use default for Smart Model
            if (selectedModel && data.models.some(m => m.name === selectedModel)) {
                modelSelect.value = selectedModel;
            } else {
                modelSelect.value = data.default;
                selectedModel = data.default;
            }

            // Restore previous selection or use default for Fast Model
             if (selectedFastModel && data.models.some(m => m.name === selectedFastModel)) {
                fastModelSelect.value = selectedFastModel;
            } else {
                // Use default_fast from API if available, else fallback to main default
                const defaultFast = data.default_fast || data.default;
                fastModelSelect.value = defaultFast;
                selectedFastModel = defaultFast;
            }

        } else {
            modelSelect.innerHTML = '<option value="">No models available</option>';
            fastModelSelect.innerHTML = '<option value="">No models available</option>';
        }

    } catch (e) {
        console.error("Failed to fetch models:", e);
        modelSelect.innerHTML = '<option value="">Error loading models</option>';
        fastModelSelect.innerHTML = '<option value="">Error loading models</option>';
    }
}

// Save model selection on change
modelSelect.addEventListener('change', () => {
    selectedModel = modelSelect.value;
    localStorage.setItem('selectedModel', selectedModel);
});

fastModelSelect.addEventListener('change', () => {
    selectedFastModel = fastModelSelect.value;
    localStorage.setItem('selectedFastModel', selectedFastModel);
});


// Fetch models on page load
fetchModels();

// Auto-activate button & resize input
queryInput.addEventListener('input', function () {
    submitBtn.disabled = !this.value.trim();
    this.style.height = 'auto';
    this.style.height = (this.scrollHeight) + 'px';
});

// Enter to submit
queryInput.addEventListener('keydown', function (e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        if (!submitBtn.disabled) {
            queryForm.requestSubmit();
        }
    }
});


// Helper: Create message element
function createMessageElement(role, content = '') {
    const msgDiv = document.createElement('div');
    msgDiv.className = `message ${role}`;

    if (role === 'user') {
        const bubble = document.createElement('div');
        bubble.className = 'message-bubble';
        bubble.textContent = content;
        msgDiv.appendChild(bubble);
    } else {
        // Assistant Message Structure:
        // 1. Trace Wrapper (Thinking)
        // 2. Message Bubble (Final Answer)
        // 3. Sources Panel (Hidden initially)

        // 1. Trace Wrapper
        const traceWrapper = document.createElement('div');
        traceWrapper.className = 'trace-wrapper hidden';
        traceWrapper.innerHTML = `
            <div class="trace-header" onclick="toggleTrace(this)">
                <div class="trace-title">
                    <span class="thinking-pulse"></span>
                    <span>Reasoning Process</span>
                </div>
                <svg class="chevron" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M6 9l6 6 6-6" />
                </svg>
            </div>
            <div class="trace-content"></div>
        `;
        msgDiv.appendChild(traceWrapper);

        // 2. Answer Bubble
        const bubble = document.createElement('div');
        bubble.className = 'message-bubble prose hidden';
        msgDiv.appendChild(bubble);

        // 3. Sources Panel
        const sourcesPanel = document.createElement('div');
        sourcesPanel.className = 'sources-panel hidden';
        sourcesPanel.innerHTML = `
            <div class="sources-header" onclick="toggleSources(this)">
                <div class="sources-title">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <circle cx="11" cy="11" r="8"></circle>
                        <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
                    </svg>
                    <span>Sources & Search Data</span>
                </div>
                <svg class="chevron" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M6 9l6 6 6-6" />
                </svg>
            </div>
            <div class="sources-content"></div>
        `;
        msgDiv.appendChild(sourcesPanel);
    }
    return msgDiv;
}

// Global scope for onclick handler in HTML string
window.toggleTrace = function (headerElement) {
    const content = headerElement.nextElementSibling;
    const chevron = headerElement.querySelector('.chevron');

    headerElement.classList.toggle('expanded');
    content.classList.toggle('expanded');
    chevron.classList.toggle('rotate-180');
};

window.toggleSources = function (headerElement) {
    const content = headerElement.nextElementSibling;
    const chevron = headerElement.querySelector('.chevron');

    headerElement.classList.toggle('expanded');
    content.classList.toggle('expanded');
    chevron.classList.toggle('rotate-180');
};

// Markdown configuration
marked.setOptions({
    highlight: function (code, lang) {
        // You could add a syntax highlighter here later
        return code;
    }
});

// =============================================================================
// SETTINGS LOGIC
// =============================================================================

function loadSettings() {
    const provider = localStorage.getItem('search_provider') || 'ddg';
    const tavilyKey = localStorage.getItem('search_api_key_tavily') || '';
    const exaKey = localStorage.getItem('search_api_key_exa') || '';
    const braveKey = localStorage.getItem('search_api_key_brave') || '';
    const googleKey = localStorage.getItem('search_api_key_google') || '';
    const googleCse = localStorage.getItem('search_cse_id_google') || '';

    // Sync both dropdowns
    searchProviderSelect.value = provider;
    searchProviderPill.value = provider;

    tavilyKeyInput.value = tavilyKey;
    if (exaKeyInput) exaKeyInput.value = exaKey;
    braveKeyInput.value = braveKey;
    googleKeyInput.value = googleKey;
    googleCseIdInput.value = googleCse;

    updateSettingsUI(provider);
}

function updateSettingsUI(provider) {
    tavilyConfig.classList.add('hidden');
    if (exaConfig) exaConfig.classList.add('hidden');
    braveConfig.classList.add('hidden');
    googleConfig.classList.add('hidden');

    if (provider === 'tavily') tavilyConfig.classList.remove('hidden');
    if (provider === 'exa' && exaConfig) exaConfig.classList.remove('hidden');
    if (provider === 'brave') braveConfig.classList.remove('hidden');
    if (provider === 'google') googleConfig.classList.remove('hidden');

    // Auto mode might need keys for underlying providers,
    // but for simplicity we don't show all configs.
    // Users should configure individual providers first.
}

// Handle Modal Dropdown Change
searchProviderSelect.addEventListener('change', (e) => {
    const provider = e.target.value;
    updateSettingsUI(provider);
    searchProviderPill.value = provider; // Sync Pill
    localStorage.setItem('search_provider', provider); // Auto-save selection
});

// Handle Pill Dropdown Change
searchProviderPill.addEventListener('change', (e) => {
    const provider = e.target.value;
    searchProviderSelect.value = provider; // Sync Modal
    updateSettingsUI(provider);
    localStorage.setItem('search_provider', provider); // Auto-save selection

    // Auto-prompt for key if missing
    checkKeyAndPrompt(provider);
});

function checkKeyAndPrompt(provider) {
    let missingKey = false;
    if (provider === 'tavily' && !localStorage.getItem('search_api_key_tavily')) missingKey = true;
    if (provider === 'exa' && !localStorage.getItem('search_api_key_exa')) missingKey = true;
    if (provider === 'brave' && !localStorage.getItem('search_api_key_brave')) missingKey = true;
    if (provider === 'google' && (!localStorage.getItem('search_api_key_google') || !localStorage.getItem('search_cse_id_google'))) missingKey = true;

    if (missingKey) {
        // Open modal to prompt user
        loadSettings(); // Refresh fields
        settingsModal.classList.remove('hidden');
        // Maybe highlight the specific section?
    }
}

// Open Modal Buttons
settingsBtn.addEventListener('click', () => {
    loadSettings();
    settingsModal.classList.remove('hidden');
});

if (quickSettingsBtn) {
    quickSettingsBtn.addEventListener('click', () => {
        loadSettings();
        settingsModal.classList.remove('hidden');
    });
}

closeSettings.addEventListener('click', () => {
    settingsModal.classList.add('hidden');
});

// Close on backdrop click
settingsModal.addEventListener('click', (e) => {
    if (e.target === settingsModal) {
        settingsModal.classList.add('hidden');
    }
});

saveSettingsBtn.addEventListener('click', () => {
    const provider = searchProviderSelect.value;
    localStorage.setItem('search_provider', provider);

    if (tavilyKeyInput.value) localStorage.setItem('search_api_key_tavily', tavilyKeyInput.value);
    if (exaKeyInput && exaKeyInput.value) localStorage.setItem('search_api_key_exa', exaKeyInput.value);
    if (braveKeyInput.value) localStorage.setItem('search_api_key_brave', braveKeyInput.value);
    if (googleKeyInput.value) localStorage.setItem('search_api_key_google', googleKeyInput.value);
    if (googleCseIdInput.value) localStorage.setItem('search_cse_id_google', googleCseIdInput.value);

    // Sync pill again just in case
    searchProviderPill.value = provider;

    settingsModal.classList.add('hidden');
});

function getSearchConfig() {
    const provider = localStorage.getItem('search_provider') || 'ddg';
    let apiKey = null;
    let cseId = null;

    if (provider === 'tavily') apiKey = localStorage.getItem('search_api_key_tavily');
    if (provider === 'exa') apiKey = localStorage.getItem('search_api_key_exa');
    if (provider === 'brave') apiKey = localStorage.getItem('search_api_key_brave');
    if (provider === 'google') {
        apiKey = localStorage.getItem('search_api_key_google');
        cseId = localStorage.getItem('search_cse_id_google');
    }

    // For Auto mode, we might want to send ALL keys?
    // Currently the backend only accepts one 'search_api_key'.
    // To support Auto mode fully, we should ideally send all keys.
    // However, for MVP, we'll assume the backend can pick up keys from Environment Variables
    // OR we need to update the request schema to support multiple keys.
    //
    // HACK: For 'auto', we will NOT send a specific key here, relying on Backend Env Vars
    // OR the user must manually switch to the specific provider to set the key.
    //
    // Correction: We should allow sending a map of keys.
    // But `ReasoningRequest` schema is strict.

    return { provider, apiKey, cseId };
}

// Initial Load
loadSettings();


// Submit handler
queryForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const query = queryInput.value.trim();
    if (!query) return;

    // Reset UI for stream
    placeholder.classList.add('hidden');
    submitBtn.disabled = true;
    queryInput.value = '';
    queryInput.style.height = 'auto';

    // Create AbortController for this request
    const controller = new AbortController();
    const signal = controller.signal;

    // Create Stop Button UI
    const stopBtn = document.createElement('button');
    stopBtn.type = 'button';
    stopBtn.className = 'stop-btn';
    stopBtn.innerHTML = `
        <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
            <rect x="6" y="6" width="12" height="12" rx="2" />
        </svg>
        Stop Generation
    `;
    stopBtn.onclick = () => {
        stopBtn.disabled = true;
        stopBtn.innerHTML = 'Stopping...';
        controller.abort();
    };

    // Swap Submit for Stop Button in the same position
    const parent = submitBtn.parentNode;
    submitBtn.classList.add('hidden');
    parent.insertBefore(stopBtn, submitBtn);


    // 1. Append User Message
    const userMsg = createMessageElement('user', query);
    chatContainer.appendChild(userMsg);
    // Scroll to bottom
    window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' });

    // 2. Append Assistant Placeholder
    const assistantMsg = createMessageElement('assistant');
    chatContainer.appendChild(assistantMsg);

    const traceWrapper = assistantMsg.querySelector('.trace-wrapper');
    const traceContent = assistantMsg.querySelector('.trace-content');
    const answerBubble = assistantMsg.querySelector('.message-bubble');
    const sourcesPanel = assistantMsg.querySelector('.sources-panel'); // New
    const sourcesContent = assistantMsg.querySelector('.sources-content'); // New
    const pulse = assistantMsg.querySelector('.thinking-pulse');

    // Helper: Mark trace as finished
    function setTraceComplete(status = 'success') {
        traceWrapper.classList.add('complete');
        const traceHeader = traceWrapper.querySelector('.trace-header');
        traceHeader.classList.add('complete');
        const traceTitleSpan = traceHeader.querySelector('.trace-title span:last-child');

        if (status === 'success') {
            if (traceTitleSpan) traceTitleSpan.textContent = 'Reasoning Completed';
            pulse.style.background = '#10b981';
        } else if (status === 'stopped') {
            if (traceTitleSpan) traceTitleSpan.textContent = 'Reasoning Stopped';
            pulse.style.background = '#fbbf24'; // Amber
        } else {
            if (traceTitleSpan) traceTitleSpan.textContent = 'Reasoning Failed';
            pulse.style.background = '#ef4444'; // Red
        }

        pulse.style.opacity = '1';
        pulse.style.animation = 'none';
        pulse.style.boxShadow = 'none';
    }

    // Show initial processing state
    const processingStatus = document.createElement('div');
    processingStatus.className = 'processing-status';
    processingStatus.innerHTML = `<span>Initializing System 2</span><span class="processing-dots"></span>`;
    traceContent.appendChild(processingStatus);

    // Ensure trace is visible for status
    if (showTraceCheckbox.checked) {
        traceWrapper.classList.remove('hidden');
        if (!traceWrapper.querySelector('.trace-header').classList.contains('expanded')) {
            toggleTrace(traceWrapper.querySelector('.trace-header'));
        }
    }

    // Parser State
    let buffer = ''; // Raw stream buffer
    let tokenBuffer = ''; // Buffer for tag detection
    let isInThinkTag = false;
    let reasoningAccumulator = '';
    let answerAccumulator = '';
    let currentNode = null;
    let stepCount = 0;
    let firstChunk = true;

    // SSE Parser State
    let currentEvent = null;
    let currentData = null;

    try {
        const searchConfig = getSearchConfig();

        // Collect all available keys for the router
        const apiKeys = {};
        if (localStorage.getItem('search_api_key_tavily')) apiKeys['tavily'] = localStorage.getItem('search_api_key_tavily');
        if (localStorage.getItem('search_api_key_exa')) apiKeys['exa'] = localStorage.getItem('search_api_key_exa');
        if (localStorage.getItem('search_api_key_brave')) apiKeys['brave'] = localStorage.getItem('search_api_key_brave');
        if (localStorage.getItem('search_api_key_google')) apiKeys['google'] = localStorage.getItem('search_api_key_google');

        const response = await fetchWithRetry('/v1/reason/stream', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                query: query,
                model: selectedModel,
                fast_model: selectedFastModel, // Send fast model
                max_iterations: 5,
                history: chatHistory,
                search_provider: searchConfig.provider,
                search_api_key: searchConfig.apiKey,
                search_cse_id: searchConfig.cseId,
                search_api_keys: apiKeys
            }),
            signal: signal
        });

        const reader = response.body.getReader();
        const decoder = new TextDecoder();

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            const text = decoder.decode(value, { stream: true });
            buffer += text;
            const lines = buffer.split('\n');
            buffer = lines.pop(); // Keep incomplete line

            for (const line of lines) {
                const trimmed = line.trim();

                if (trimmed === '') {
                    // --- Dispatch Event (End of Block) ---
                    if (currentData) {

                        // 1. Handle Done Signal
                        if (currentEvent === 'done') {
                            // ...
                        }

                        // 2. Handle Ping (keep-alive)
                        else if (currentEvent === 'ping') {
                            // Do nothing, just keep-alive
                        }

                        // 3. Handle Token Streaming
                        else if (currentEvent === 'tool_io') {
                            try {
                                const toolData = JSON.parse(currentData);
                                console.log("[app.js] tool_io event:", toolData);

                                if (toolData.type === 'search_result') {
                                    // Make Sources Panel Visible
                                    sourcesPanel.classList.remove('hidden');
                                    if (sourcesContent.parentNode.classList.contains('hidden')) {
                                        sourcesContent.parentNode.classList.remove('hidden');
                                    }

                                    // Create Search Card
                                    const card = document.createElement('div');
                                    card.className = 'search-card';

                                    const provider = toolData.provider || 'unknown';
                                    const query = toolData.query || 'Unknown Query';
                                    const count = toolData.count || 0;
                                    const results = toolData.results || [];

                                    let resultsHtml = '';
                                    results.forEach(res => {
                                        resultsHtml += `<li class="search-result-item"><a href="${res.url}" target="_blank" title="${res.title}">${res.title}</a></li>`;
                                    });

                                    card.innerHTML = `
                                        <div class="search-card-header">
                                            <span class="provider-badge ${provider}">${provider}</span>
                                            <span style="font-size: 0.7rem; color: #71717a;">${count} results</span>
                                        </div>
                                        <div class="search-query">"${query}"</div>
                                        <ul class="search-results-list">
                                            ${resultsHtml}
                                        </ul>
                                    `;

                                    sourcesContent.appendChild(card);
                                }
                            } catch (e) { console.warn("Tool IO Parse Error:", e); }
                        }

                        else {
                            try {
                                const data = JSON.parse(currentData);
                                const token = data.token;
                                if (token) {
                                    const upstreamNode = data.node || 'reason';

                                    // Handle Node Transition
                                    if (upstreamNode !== currentNode) {
                                        if (currentNode && upstreamNode === 'reason' && answerAccumulator) {
                                            // Archive Draft Logic
                                            const draftDiv = document.createElement('div');
                                            draftDiv.className = 'draft-answer-block';
                                            draftDiv.innerHTML = `<span class="draft-label">Draft Answer #${stepCount}</span>${marked.parse(answerAccumulator)}`;
                                            traceContent.appendChild(draftDiv);

                                            // Reset for new attempt
                                            answerAccumulator = '';
                                            answerBubble.innerHTML = '';
                                            answerBubble.classList.add('hidden');
                                        }

                                        currentNode = upstreamNode;
                                        stepCount++;

                                        if (firstChunk) {
                                            processingStatus.remove();
                                            firstChunk = false;
                                        }

                                        const header = document.createElement('div');
                                        header.className = 'trace-step-header';
                                        header.textContent = `Step ${stepCount}: ${currentNode}`;
                                        traceContent.appendChild(header);
                                    }

                                    // --- Robust Parsing Logic ---
                                    tokenBuffer += token;

                                    // Define which nodes contribute ONLY to trace (no answer accumulation)
                                    // CRITICAL: Exclude 'mcts_final' so it can populate the Answer Bubble!
                                    const isTraceOnlyNode = ['plan', 'tool', 'critique', 'mcts_select', 'mcts_expand', 'mcts_evaluate', 'mcts_backprop'].some(p => currentNode && currentNode.startsWith(p));

                                    while (true) {
                                        const thinkStart = tokenBuffer.indexOf('<think>');
                                        const thinkEnd = tokenBuffer.indexOf('</think>');

                                        if (isInThinkTag) {
                                            if (thinkEnd !== -1) {
                                                const content = tokenBuffer.slice(0, thinkEnd);
                                                reasoningAccumulator += content;
                                                traceContent.appendChild(document.createTextNode(content));
                                                tokenBuffer = tokenBuffer.slice(thinkEnd + 8);
                                                isInThinkTag = false;

                                                // Update UI
                                                traceWrapper.classList.add('complete');
                                            } else {
                                                // No closing tag yet - flush safe content
                                                let safeIndex = tokenBuffer.length;
                                                // Keep last 7 chars in buffer to avoiding splitting </think>
                                                if (tokenBuffer.length > 7) {
                                                    safeIndex = tokenBuffer.length - 7;
                                                } else {
                                                    safeIndex = 0; // Don't flush if too short
                                                }

                                                if (safeIndex > 0) {
                                                    const safeContent = tokenBuffer.slice(0, safeIndex);
                                                    reasoningAccumulator += safeContent;

                                                    // Render search tags
                                                    const searchRender = safeContent.replace(/<search>(.*?)<\/search>/gs, '<div class="search-pill"><span class="search-icon">🔍</span> $1</div>');

                                                    if (currentNode === 'tool') {
                                                        const toolContent = document.createElement('div');
                                                        toolContent.className = 'search-results-block';
                                                        toolContent.textContent = safeContent;
                                                        traceContent.appendChild(toolContent);
                                                    } else {
                                                        const span = document.createElement('span');
                                                        span.innerHTML = searchRender;
                                                        traceContent.appendChild(span);
                                                    }

                                                    tokenBuffer = tokenBuffer.slice(safeIndex);
                                                }
                                                break; // Wait for more tokens
                                            }
                                        } else {
                                            if (thinkStart !== -1) {
                                                const content = tokenBuffer.slice(0, thinkStart);

                                                if (isTraceOnlyNode) {
                                                    reasoningAccumulator += content;
                                                    traceContent.appendChild(document.createTextNode(content));
                                                } else {
                                                    answerAccumulator += content;
                                                }

                                                tokenBuffer = tokenBuffer.slice(thinkStart + 7);
                                                isInThinkTag = true;
                                                traceWrapper.classList.remove('hidden');
                                                traceWrapper.classList.remove('complete');
                                            } else {
                                                // No think tag start
                                                if (isTraceOnlyNode) {
                                                    // Everything goes to trace for these nodes
                                                    reasoningAccumulator += tokenBuffer;
                                                    // Special handling for search pill rendering in trace-only nodes too
                                                    const searchRender = tokenBuffer.replace(/<search>(.*?)<\/search>/gs, '<div class="search-pill"><span class="search-icon">🔍</span> $1</div>');
                                                    if (searchRender !== tokenBuffer) {
                                                        const span = document.createElement('span');
                                                        span.innerHTML = searchRender;
                                                        traceContent.appendChild(span);
                                                    } else {
                                                        traceContent.appendChild(document.createTextNode(tokenBuffer));
                                                    }
                                                } else {
                                                    answerAccumulator += tokenBuffer;
                                                }
                                                tokenBuffer = '';
                                                break;
                                            }
                                        }
                                    }
                                }
                            } catch (e) { console.warn("Parse Error:", e); }
                        }
                    } // End dispatch

                    if (currentEvent === 'done') break; // Break for lines loop

                    // Reset for next event
                    currentEvent = null;
                    currentData = null;

                } else if (line.startsWith('event: ')) {
                    currentEvent = line.slice(7).trim();
                } else if (line.startsWith('data: ')) {
                    if (currentData === null) currentData = '';
                    currentData += line.slice(6) + '\n';
                }
            }

            requestAnimationFrame(() => {
                if (traceContent.classList.contains('expanded')) {
                    traceContent.scrollTop = traceContent.scrollHeight;
                }
                if (answerAccumulator) {
                    answerBubble.classList.remove('hidden');
                    // Pre-process LaTeX: Wrap \boxed{} in $$ delimiters for MathJax
                    // Simple regex for basic nesting support
                    const latexAnswer = answerAccumulator.replace(/(\\boxed\{((?:[^{}]+|\{[^{}]*\})*)\})/g, '$$$1$$');
                    answerBubble.innerHTML = marked.parse(latexAnswer);
                }
                const isAtBottom = (window.innerHeight + window.scrollY) >= document.body.offsetHeight - 100;
                if (isAtBottom) {
                    window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' });
                }
            });

            if (currentEvent === 'done') break; // Break outer loop
        }

        // Finalize turn (Success)
        setTraceComplete('success');

        // Add Final Answer Badge
        if (answerBubble && !answerBubble.classList.contains('hidden')) {
            const badge = document.createElement('div');
            badge.className = 'final-answer-badge';
            badge.textContent = 'Final Answer';
            answerBubble.insertBefore(badge, answerBubble.firstChild);
        }

        // Trigger MathJax
        if (window.MathJax) {
            try {
                await window.MathJax.typesetPromise([answerBubble]);
            } catch (e) {
                console.warn("MathJax error:", e);
                // Retry once in case of race condition
                setTimeout(() => window.MathJax.typesetPromise([answerBubble]), 500);
            }
        }

        chatHistory.push({ role: 'user', content: query });
        chatHistory.push({ role: 'assistant', content: answerAccumulator });

    } catch (error) {
        console.error("Stream error:", error);
        if (error.name === 'AbortError') {
            setTraceComplete('stopped');
            const cancelledMsg = document.createElement('div');
            cancelledMsg.className = 'error-badge';
            cancelledMsg.style.color = '#fbbf24';
            cancelledMsg.style.marginTop = '1rem';
            cancelledMsg.textContent = 'Generation stopped by user.';
            traceContent.appendChild(cancelledMsg);
        } else {
            setTraceComplete('error');
            answerBubble.innerHTML = `<div style="color: #ef4444; padding: 1rem; border: 1px solid #ef4444; border-radius: 8px;">Connection Error: ${error.message}. Please check if Ollama is running.</div>`;
            answerBubble.classList.remove('hidden');
        }
    } finally {
        // cleanup
        try {
            if (firstChunk && processingStatus && processingStatus.parentNode) {
                processingStatus.remove();
            }
        } catch (e) { console.warn("Error removing status:", e); }

        submitBtn.disabled = false;
        submitBtn.classList.remove('hidden');

        try {
            if (stopBtn && stopBtn.parentNode) {
                stopBtn.remove();
            }
        } catch (e) { console.warn("Error removing stop btn:", e); }

        queryInput.focus();
    }
});
