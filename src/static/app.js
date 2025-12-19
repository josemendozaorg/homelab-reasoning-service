const queryInput = document.getElementById('queryInput');
const submitBtn = document.getElementById('submitBtn');
const queryForm = document.getElementById('queryForm');
const chatContainer = document.getElementById('chatContainer');
const placeholder = document.getElementById('placeholder');
const showTraceCheckbox = document.getElementById('showTrace');
const appVersion = document.getElementById('appVersion');

let chatHistory = []; // Local history state

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

// Keep initialization
(async () => {
    try {
        const res = await fetchWithRetry('/api/info');
        const data = await res.json();
        if (data.version) {
            appVersion.textContent = `v${data.version}`;
        }
    } catch (e) {
        console.error("Failed to fetch version:", e);
    }
})();

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

// Markdown configuration
marked.setOptions({
    highlight: function (code, lang) {
        // You could add a syntax highlighter here later
        return code;
    }
});

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

    try {
        const response = await fetchWithRetry('/v1/reason/stream', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                query: query,
                max_iterations: 5,
                history: chatHistory
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
                if (line.startsWith('data: ')) {
                    const dataStr = line.slice(6);
                    if (dataStr === 'Reasoning complete') break;

                    try {
                        const data = JSON.parse(dataStr);
                        const token = data.token;
                        const upstreamNode = data.node || 'reason';

                        // Handle Node Transition
                        if (upstreamNode !== currentNode) {
                            if (currentNode && upstreamNode === 'reason' && answerAccumulator) {
                                // Archive Draft
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

                        // --- Robust Tag Parsing Logic ---
                        tokenBuffer += token;

                        // Process buffer while it contains closed tags or is safe to flush
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
                                } else {
                                    // Robust wait for close tag
                                    let safeIndex = tokenBuffer.length;
                                    for (let i = 1; i <= 7; i++) {
                                        if (tokenBuffer.endsWith("</think>".slice(0, i))) {
                                            safeIndex = tokenBuffer.length - i;
                                            break;
                                        }
                                    }

                                    if (safeIndex < tokenBuffer.length) {
                                        const safeContent = tokenBuffer.slice(0, safeIndex);
                                        reasoningAccumulator += safeContent;

                                        // Simple regex replace for visibility (better would be to parse it out completely)
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
                                        break;
                                    } else {
                                        reasoningAccumulator += tokenBuffer;
                                        const searchRenderAll = tokenBuffer.replace(/<search>(.*?)<\/search>/gs, '<div class="search-pill"><span class="search-icon">🔍</span> $1</div>');
                                        if (currentNode === 'tool') {
                                            const toolContent = document.createElement('div');
                                            toolContent.className = 'search-results-block';
                                            toolContent.textContent = tokenBuffer;
                                            traceContent.appendChild(toolContent);
                                        } else {
                                            const span = document.createElement('span');
                                            span.innerHTML = searchRenderAll;
                                            traceContent.appendChild(span);
                                        }
                                        tokenBuffer = '';
                                        break;
                                    }
                                }
                            } else {
                                if (thinkStart !== -1) {
                                    const content = tokenBuffer.slice(0, thinkStart);
                                    if (currentNode === 'critique') {
                                        reasoningAccumulator += content;
                                        traceContent.appendChild(document.createTextNode(content));
                                    } else {
                                        answerAccumulator += content;
                                    }
                                    tokenBuffer = tokenBuffer.slice(thinkStart + 7);
                                    isInThinkTag = true;
                                    traceWrapper.classList.remove('hidden');
                                } else {
                                    let safeIndex = tokenBuffer.length;
                                    for (let i = 1; i <= 6; i++) {
                                        if (tokenBuffer.endsWith("<think>".slice(0, i))) {
                                            safeIndex = tokenBuffer.length - i;
                                            break;
                                        }
                                    }
                                    if (safeIndex < tokenBuffer.length) {
                                        const safeContent = tokenBuffer.slice(0, safeIndex);
                                        if (currentNode === 'critique') {
                                            reasoningAccumulator += safeContent;
                                            traceContent.appendChild(document.createTextNode(safeContent));
                                        } else {
                                            answerAccumulator += safeContent;
                                        }
                                        tokenBuffer = tokenBuffer.slice(safeIndex);
                                        break;
                                    } else {
                                        if (currentNode === 'critique') {
                                            reasoningAccumulator += tokenBuffer;
                                            traceContent.appendChild(document.createTextNode(tokenBuffer));
                                        } else {
                                            answerAccumulator += tokenBuffer;
                                        }
                                        tokenBuffer = '';
                                        break;
                                    }
                                }
                            }
                        }

                        requestAnimationFrame(() => {
                            if (traceContent.classList.contains('expanded')) {
                                traceContent.scrollTop = traceContent.scrollHeight;
                            }
                            if (answerAccumulator) {
                                answerBubble.classList.remove('hidden');
                                answerBubble.innerHTML = marked.parse(answerAccumulator);
                            }
                            const isAtBottom = (window.innerHeight + window.scrollY) >= document.body.offsetHeight - 100;
                            if (isAtBottom) {
                                window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' });
                            }
                        });


                    } catch (e) {
                        // ignore
                    }
                }
            }
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
