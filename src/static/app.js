const queryInput = document.getElementById('queryInput');
const submitBtn = document.getElementById('submitBtn');
const queryForm = document.getElementById('queryForm');
const chatContainer = document.getElementById('chatContainer');
const placeholder = document.getElementById('placeholder');
const showTraceCheckbox = document.getElementById('showTrace');
const appVersion = document.getElementById('appVersion');

let chatHistory = []; // Local history state

// Initialize
(async () => {
    try {
        const res = await fetch('/api/info');
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

    // Show thinking if enabled
    if (showTraceCheckbox.checked) {
        // We will unhide it when data comes in
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

    try {
        const response = await fetch('/v1/reason/stream', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                query: query,
                max_iterations: 5,
                history: chatHistory
            })
        });

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        // State for parsing
        let isInThinkTag = false;
        let reasoningAccumulator = '';
        let answerAccumulator = '';
        let currentNode = null;
        let stepCount = 0;

        // Remove processing status on first real data
        let firstChunk = true;

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
                        const upstreamNode = data.node || 'reason'; // Default to reason if missing

                        // Handle Node Transition / Iteration Headers
                        if (upstreamNode !== currentNode) {
                            currentNode = upstreamNode;
                            stepCount++;

                            // Remove processing status if it exists
                            if (firstChunk) {
                                processingStatus.remove();
                                firstChunk = false;
                            }

                            // Inject Header into Trace
                            // Use raw HTML append to avoid overwriting content with textContent
                            const header = document.createElement('div');
                            header.className = 'trace-step-header';
                            header.textContent = `Step ${stepCount}: ${currentNode}`;
                            traceContent.appendChild(header);
                        }

                        // State machine for <think> tags
                        if (token.includes('<think>')) {
                            isInThinkTag = true;
                            traceWrapper.classList.remove('hidden');

                            const parts = token.split('<think>');
                            reasoningAccumulator += parts[1] || '';

                            // Direct append to trace content
                            if (parts[1]) {
                                traceContent.appendChild(document.createTextNode(parts[1]));
                            }

                        } else if (token.includes('</think>')) {
                            const parts = token.split('</think>');
                            reasoningAccumulator += parts[0] || '';
                            answerAccumulator += parts[1] || '';
                            isInThinkTag = false;

                            if (parts[0]) {
                                traceContent.appendChild(document.createTextNode(parts[0]));
                            }

                            // Pulse animation stops or turns green/done?
                            // For now, let's just leave it as indicator of "past thought"
                            pulse.style.opacity = '0.5';
                            pulse.style.animation = 'none';

                        } else {
                            if (isInThinkTag) {
                                reasoningAccumulator += token;
                                traceContent.appendChild(document.createTextNode(token));
                            } else {
                                answerAccumulator += token;
                            }
                        }

                        // Update DOM
                        requestAnimationFrame(() => {
                            // Scroll trace if expanded
                            if (traceContent.classList.contains('expanded')) {
                                traceContent.scrollTop = traceContent.scrollHeight;
                            }

                            if (answerAccumulator) {
                                answerBubble.classList.remove('hidden');
                                // Note: re-rendering markdown here is expensive but necessary for streaming markdown
                                answerBubble.innerHTML = marked.parse(answerAccumulator);
                            }

                            // Smooth scroll page to follow output
                            // Only if near bottom?
                            const isAtBottom = (window.innerHeight + window.scrollY) >= document.body.offsetHeight - 100;
                            if (isAtBottom) {
                                window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' });
                            }
                        });

                    } catch (e) {
                        // Ignore parse errors on chunks
                    }
                }
            }
        }

        // Finalize turn
        pulse.style.opacity = '0.2';
        pulse.style.animation = 'none'; // Stop pulsing

        chatHistory.push({ role: 'user', content: query });
        chatHistory.push({ role: 'assistant', content: answerAccumulator });

    } catch (error) {
        answerBubble.innerHTML = `<div style="color: #ef4444; padding: 1rem; border: 1px solid #ef4444; border-radius: 8px;">Encryption Error: ${error.message}</div>`;
        answerBubble.classList.remove('hidden');
    } finally {
        submitBtn.disabled = false;
        queryInput.focus();
    }
});
