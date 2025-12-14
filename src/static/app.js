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

// Auto-activate button
queryInput.addEventListener('input', () => {
    submitBtn.disabled = !queryInput.value.trim();
});

// Helper: Create message element
function createMessageElement(role, content = '') {
    const msgDiv = document.createElement('div');
    msgDiv.className = `message ${role}`;

    // User message is simple bubble
    if (role === 'user') {
        const bubble = document.createElement('div');
        bubble.className = 'message-bubble';
        bubble.textContent = content;
        msgDiv.appendChild(bubble);
    } else {
        // Assistant message has thinking (trace) + bubble (answer)

        // 1. Thinking container
        const thinkingDiv = document.createElement('div');
        thinkingDiv.className = 'thinking-container hidden'; // Hidden initially if empty
        thinkingDiv.innerHTML = `
            <div class="thinking-header" onclick="this.parentElement.classList.toggle('collapsed')">
                <span class="spinner"></span>
                <span>Reasoning Trace</span>
                <svg class="chevron" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M6 9l6 6 6-6" />
                </svg>
            </div>
            <div class="thinking-content"></div>
        `;
        msgDiv.appendChild(thinkingDiv);

        // 2. Answer bubble
        const bubble = document.createElement('div');
        bubble.className = 'message-bubble hidden'; // Hidden initially
        msgDiv.appendChild(bubble);
    }

    return msgDiv;
}

// Submit handler
queryForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const query = queryInput.value.trim();
    if (!query) return;

    // UI Updates
    placeholder.classList.add('hidden');
    submitBtn.disabled = true;
    queryInput.value = '';
    queryInput.style.height = 'auto';

    // 1. Append User Message
    const userMsg = createMessageElement('user', query);
    chatContainer.appendChild(userMsg);
    chatContainer.scrollTop = chatContainer.scrollHeight;

    // 2. Append Assistant Placeholder
    const assistantMsg = createMessageElement('assistant');
    chatContainer.appendChild(assistantMsg);

    const thinkingContainer = assistantMsg.querySelector('.thinking-container');
    const thinkingContent = assistantMsg.querySelector('.thinking-content');
    const answerBubble = assistantMsg.querySelector('.message-bubble');
    const spinner = assistantMsg.querySelector('.spinner');

    // Show thinking if enabled
    if (showTraceCheckbox.checked) {
        thinkingContainer.classList.remove('hidden');
    } else {
        thinkingContainer.classList.add('collapsed'); // Pre-collapse but show header? Or hide? 
        // Logic: if checkbox off, hide logic entirely? Or just collapse?
        // Let's hide initially, show only if trace arrives? 
        // For simplicity: always remove hidden if trace starts arriving.
        thinkingContainer.classList.remove('hidden');
        thinkingContainer.classList.add('collapsed');
    }

    try {
        const response = await fetch('/v1/reason/stream', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                query: query,
                max_iterations: 5,
                history: chatHistory // Send context
            })
        });

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        // State for parsing
        let isInThinkTag = false;
        let reasoningAccumulator = '';
        let answerAccumulator = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            const text = decoder.decode(value);
            buffer += text;
            const lines = buffer.split('\n');
            buffer = lines.pop();

            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    const dataStr = line.slice(6);
                    if (dataStr === 'Reasoning complete') break;

                    try {
                        const data = JSON.parse(dataStr);
                        const token = data.token;

                        // State machine
                        if (token.includes('<think>')) {
                            isInThinkTag = true;
                            // Ensure container visible when thought starts
                            thinkingContainer.classList.remove('hidden');
                            chatContainer.scrollTop = chatContainer.scrollHeight;

                            const parts = token.split('<think>');
                            reasoningAccumulator += parts[1] || '';
                        } else if (token.includes('</think>')) {
                            const parts = token.split('</think>');
                            reasoningAccumulator += parts[0] || '';
                            answerAccumulator += parts[1] || '';
                            isInThinkTag = false;
                        } else {
                            if (isInThinkTag) {
                                reasoningAccumulator += token;
                            } else {
                                answerAccumulator += token;
                            }
                        }

                        // Update DOM
                        requestAnimationFrame(() => {
                            if (reasoningAccumulator) thinkingContent.textContent = reasoningAccumulator;

                            if (answerAccumulator) {
                                answerBubble.innerHTML = marked.parse(answerAccumulator);
                                answerBubble.classList.remove('hidden');
                            }

                            // Auto-scroll logic (basic)
                            chatContainer.scrollTop = chatContainer.scrollHeight;
                        });

                    } catch (e) {
                        // Ignore parse errors on chunks
                    }
                }
            }
        }

        // Finalize turn
        spinner.style.display = 'none';
        chatHistory.push({ role: 'user', content: query });
        chatHistory.push({ role: 'assistant', content: answerAccumulator });

    } catch (error) {
        answerBubble.innerHTML = `<div style="color: #ef4444">Error: ${error.message}</div>`;
        answerBubble.classList.remove('hidden');
    } finally {
        submitBtn.disabled = false;
    }
});
