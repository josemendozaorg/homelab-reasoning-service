const queryInput = document.getElementById('queryInput');
const submitBtn = document.getElementById('submitBtn');
const queryForm = document.getElementById('queryForm');
const responseArea = document.getElementById('responseArea');
const thinking = document.getElementById('thinking');
const thinkingContent = document.getElementById('thinkingContent');
const finalAnswer = document.getElementById('finalAnswer');
const showTraceCheckbox = document.getElementById('showTrace');
const appVersion = document.getElementById('appVersion');

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

// Submit handler
queryForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const query = queryInput.value.trim();
    if (!query) return;

    // Reset UI
    document.querySelector('.placeholder-text').classList.add('hidden');
    thinking.classList.remove('hidden');
    thinkingContent.textContent = "Initializing reasoner...";
    finalAnswer.classList.add('hidden');
    finalAnswer.innerHTML = '';
    submitBtn.disabled = true;

    if (!showTraceCheckbox.checked) {
        thinking.classList.add('collapsed');
    }

    try {
        // Use text/event-stream for streaming
        const response = await fetch('/v1/reason/stream', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query: query, max_iterations: 5 })
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

            // SSE lines handling
            const lines = buffer.split('\n');
            buffer = lines.pop(); // Keep incomplete line

            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    const dataStr = line.slice(6);
                    if (dataStr === 'Reasoning complete') break;

                    try {
                        const data = JSON.parse(dataStr);
                        const token = data.token;

                        // Simple state machine for parsing <think> tags on the fly
                        if (token.includes('<think>')) {
                            isInThinkTag = true;
                            // Clean token
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

                        // Update UI
                        requestAnimationFrame(() => {
                            if (reasoningAccumulator) thinkingContent.textContent = reasoningAccumulator;
                            // Use marked.parse for markdown, but throttle or sanitize if heavy? 
                            // For token stream, simple innerHTML update with marked might flicker unclosed tags.
                            // For now, simpler:
                            finalAnswer.innerHTML = marked.parse(answerAccumulator);
                            finalAnswer.classList.remove('hidden');
                        });

                    } catch (e) {
                        console.error("Error parsing SSE data:", e);
                    }
                }
            }
        }

        document.querySelector('.spinner').style.display = 'none';

    } catch (error) {
        finalAnswer.innerHTML = `<div style="color: #ef4444">Network Error: ${error.message}</div>`;
        finalAnswer.classList.remove('hidden');
    } finally {
        submitBtn.disabled = false;
        queryInput.value = '';
        queryInput.style.height = 'auto'; // Reset height
    }
});

function toggleThinking() {
    thinking.classList.toggle('collapsed');
}
