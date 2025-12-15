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

        // Parser State
        let buffer = ''; // Raw stream buffer
        let tokenBuffer = ''; // Buffer for tag detection
        let isInThinkTag = false;
        let reasoningAccumulator = '';
        let answerAccumulator = '';
        let currentNode = null;
        let stepCount = 0;
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
                        const upstreamNode = data.node || 'reason';

                        // Handle Node Transition
                        if (upstreamNode !== currentNode) {
                            // If we are switching back to 'reason' from something else (or initial)
                            // AND we already have an answerAccumulator (meaning we have a draft),
                            // we need to archive the current draft to the trace.
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
                                    // Found closing tag: flush content up to tag, switch state
                                    const content = tokenBuffer.slice(0, thinkEnd);
                                    reasoningAccumulator += content;
                                    traceContent.appendChild(document.createTextNode(content));

                                    // Remove processed part + tag
                                    tokenBuffer = tokenBuffer.slice(thinkEnd + 8); // 8 is len of </think>
                                    isInThinkTag = false;
                                } else {
                                    // No closing tag yet. 
                                    // Check for PARTIAL closing tag at the end (e.g. "</", "</th")
                                    // If partial tag exists, wait for more data.
                                    // Otherwise, flush everything.

                                    // Regex for partial ending of </think>
                                    // potential suffixes: <, </, </t, </th, </thi, </thin, </think (handled above)
                                    const partialMatch = tokenBuffer.match(/(<|\/|< \/| <\/t|<\/th|<\/thi|<\/thin|<\/think)$/); // Simplified check

                                    // More robust check: does string end with prefix of "</think>"?
                                    let safeIndex = tokenBuffer.length;
                                    for (let i = 1; i <= 7; i++) {
                                        if (tokenBuffer.endsWith("</think>".slice(0, i))) {
                                            safeIndex = tokenBuffer.length - i;
                                            break;
                                        }
                                    }

                                    if (safeIndex < tokenBuffer.length) {
                                        // Flush up to safeIndex
                                        const safeContent = tokenBuffer.slice(0, safeIndex);
                                        reasoningAccumulator += safeContent;
                                        traceContent.appendChild(document.createTextNode(safeContent));
                                        tokenBuffer = tokenBuffer.slice(safeIndex);
                                        break; // Need more data to resolve tag
                                    } else {
                                        // Safe to flush all
                                        reasoningAccumulator += tokenBuffer;
                                        traceContent.appendChild(document.createTextNode(tokenBuffer));
                                        tokenBuffer = '';
                                        break; // Done with this batch
                                    }
                                }
                            } else {
                                // Not in think tag
                                if (thinkStart !== -1) {
                                    // Found opening tag
                                    // Flush content before tag to answer (unless node is critique)
                                    const content = tokenBuffer.slice(0, thinkStart);

                                    if (currentNode === 'critique') {
                                        reasoningAccumulator += content;
                                        traceContent.appendChild(document.createTextNode(content));
                                    } else {
                                        answerAccumulator += content;
                                    }

                                    tokenBuffer = tokenBuffer.slice(thinkStart + 7); // 7 is len of <think>
                                    isInThinkTag = true;
                                    traceWrapper.classList.remove('hidden');
                                } else {
                                    // No opening tag. Check partial opening <think>
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

                        // UI Updates
                        requestAnimationFrame(() => {
                            // Scroll trace if expanding
                            if (traceContent.classList.contains('expanded')) {
                                traceContent.scrollTop = traceContent.scrollHeight;
                            }

                            if (answerAccumulator) {
                                answerBubble.classList.remove('hidden');
                                answerBubble.innerHTML = marked.parse(answerAccumulator);
                            }

                            // Smooth scroll
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

        // Finalize turn
        pulse.style.opacity = '0.5';
        pulse.style.animation = 'none'; // Stop pulsing

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
        answerBubble.innerHTML = `<div style="color: #ef4444; padding: 1rem; border: 1px solid #ef4444; border-radius: 8px;">Encryption Error: ${error.message}</div>`;
        answerBubble.classList.remove('hidden');
    } finally {
        submitBtn.disabled = false;
        queryInput.focus();
    }
});
