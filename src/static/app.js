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

    // Handle initial trace visibility preference
    if (showTraceCheckbox.checked) {
        // We will unhide it when data comes in
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

                        // State machine for <think> tags
                        if (token.includes('<think>')) {
                            isInThinkTag = true;
                            traceWrapper.classList.remove('hidden');

                            // If user wants to see trace, expand it by default on start
                            if (showTraceCheckbox.checked && !traceWrapper.querySelector('.trace-header').classList.contains('expanded')) {
                                toggleTrace(traceWrapper.querySelector('.trace-header'));
                            }

                            const parts = token.split('<think>');
                            reasoningAccumulator += parts[1] || '';

                        } else if (token.includes('</think>')) {
                            const parts = token.split('</think>');
                            reasoningAccumulator += parts[0] || '';
                            answerAccumulator += parts[1] || '';
                            isInThinkTag = false;

                            // Pulse animation stops or turns green/done? 
                            // For now, let's just leave it as indicator of "past thought"
                            pulse.style.opacity = '0.5';
                            pulse.style.animation = 'none';

                        } else {
                            if (isInThinkTag) {
                                reasoningAccumulator += token;
                            } else {
                                answerAccumulator += token;
                            }
                        }

                        // Update DOM
                        requestAnimationFrame(() => {
                            if (reasoningAccumulator) {
                                // For reasoning trace, we just append text comfortably
                                traceContent.textContent = reasoningAccumulator;
                                // Auto-scroll trace if expanded
                                if (traceContent.classList.contains('expanded')) {
                                    traceContent.scrollTop = traceContent.scrollHeight;
                                }
                            }

                            if (answerAccumulator) {
                                answerBubble.classList.remove('hidden');

                                // To achieve the "token" animation effect without breaking Markdown parsing,
                                // we need to render the Markdown but try to animate the new parts.
                                // However, re-rendering Markdown on every token destroys the DOM nodes and restarts animations.
                                // A common trick is to append raw HTML for the "live" part or use a specific container.

                                // BUT, for a smoother robust implementation that supports Markdown:
                                // We will render the FULL markdown, but the animation effect is best applied 
                                // if we can isolate what changed.

                                // COMPROMISE for "Circular Streaming" + "Readable":
                                // We will update the innerHTML with the parsed markdown.
                                // To get the "circular" feel, we can't easily wrap every character in the parsed HTML 
                                // without complex diffing.

                                // INSTED: We will apply a mask effect or just standard fade-in for the whole block? No.
                                // User wants "circular pattern".

                                // Let's try formatting the *new* text chunks as they come in? 
                                // We can't easily do that with `marked.parse(accumulated)`.

                                // ALTERNATIVE: Just update the HTML.
                                // The user asked for "show the text slowly in a circular pattern".
                                // If we just update innerHTML, it snaps in.

                                // Let's try to simulate the effect by appending spans for the *streaming* phase
                                // effectively bypassing Markdown for the "live" tail, OR just accepting that
                                // Markdown re-render kills animation.

                                // Let's stick to the Robust Method:
                                // 1. Render Markdown to a temp container.
                                // 2. Replace content.
                                // 3. For the "Circular" feel, we add a CSS class to the MAIN container that has a 
                                //    radial-gradient mask that moves? That might be too complex for this file.

                                // SIMPLIFIED: We will wrap the text in a span if we were appending raw text.
                                // Since we need Markdown, we will only animate the *Reasoning Trace* (which is raw text) easily.
                                // For the *Answer*, we will rely on a generic fade in, 
                                // UNLESS we implement a custom renderer.

                                // WAIT, the user said "result of the LLM".
                                // Let's apply the effect to the TRACE primarily, or the Answer?
                                // "Result" usually implies the Answer.

                                // OK, to support Markdown + Animation:
                                // The only clean way is to append tokens as HTML elements *until* the end, then Markdown-ify?
                                // No, because we want bold/lists to appear immediately.

                                // Let's assume we just render.
                                // For the "Circular" effect, let's update the styles in style.css to have a 
                                // @keyframes that scans across the text?
                                // No, that repeats.

                                // LET'S DO THIS:
                                // We will wrap the *new content* in a span `class="reveal-token"` 
                                // ONLY IF we are in a text-only mode or if we accept that Markdown only happens at the end?
                                // No, user wants readable.

                                // HYBRID: Render Markdown. 
                                // AND add a "scanning radar" overlay to the answerBubble?
                                // That fits "Nano Banana".
                                // Let's do that. It's a visual effect on the container.
                                answerBubble.innerHTML = marked.parse(answerAccumulator);
                                answerBubble.classList.add('scanning-active');
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
