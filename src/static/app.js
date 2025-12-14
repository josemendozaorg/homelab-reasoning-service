const queryInput = document.getElementById('queryInput');
const submitBtn = document.getElementById('submitBtn');
const queryForm = document.getElementById('queryForm');
const responseArea = document.getElementById('responseArea');
const thinking = document.getElementById('thinking');
const thinkingContent = document.getElementById('thinkingContent');
const finalAnswer = document.getElementById('finalAnswer');
const showTraceCheckbox = document.getElementById('showTrace');

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
        const response = await fetch('/v1/reason', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                query: query,
                max_iterations: 5
            })
        });

        const data = await response.json();

        if (response.ok) {
            // Render Reasoning Trace
            // If data.reasoning_trace is a list of steps, join them
            // If it's empty, use a placeholder
            if (data.reasoning_trace && data.reasoning_trace.length > 0) {
                // Format trace nicely
                const traceHtml = data.reasoning_trace.map((step, index) =>
                    `[Step ${index + 1}]\n${JSON.stringify(step, null, 2)}`
                ).join('\n\n');
                thinkingContent.textContent = traceHtml;
            } else {
                thinkingContent.textContent = "Reasoning complete (No granular trace returned).";
            }

            // Render Final Answer
            finalAnswer.innerHTML = marked.parse(data.final_answer || "**No answer returned.**");
            finalAnswer.classList.remove('hidden');

            // Stop spinner by removing the class (optional, or just logic)
            document.querySelector('.spinner').style.display = 'none';

        } else {
            finalAnswer.innerHTML = `<div style="color: #ef4444">Error: ${data.detail || 'Unknown error'}</div>`;
            finalAnswer.classList.remove('hidden');
        }

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
