// Initialize markdown-it
const md = window.markdownit({
    html: true,
    linkify: true,
    typographer: true,
    highlight: function (str, lang) {
        if (lang && Prism.languages[lang]) {
            try {
                return `<pre class="line-numbers"><code class="language-${lang}">${Prism.highlight(str, Prism.languages[lang], lang)}</code></pre>`;
            } catch (__) {}
        }
        return `<pre class="line-numbers"><code class="language-plaintext">${md.utils.escapeHtml(str)}</code></pre>`;
    }
});

// Auto-resize textarea
const textarea = document.getElementById('queryInput');
textarea.addEventListener('input', function() {
    this.style.height = 'auto';
    this.style.height = (this.scrollHeight) + 'px';
});

// Handle Enter key (Shift+Enter for new line)
function handleKeyDown(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        sendQuery();
    }
}

// Clear chat history
function clearChat() {
    const chatContainer = document.getElementById('chatContainer');
    chatContainer.innerHTML = '';
}

// Add message to chat
function addMessage(content, isUser = false) {
    const chatContainer = document.getElementById('chatContainer');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${isUser ? 'user-message' : 'assistant-message'}`;
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    
    // Process markdown and sanitize HTML
    const processedContent = DOMPurify.sanitize(md.render(content));
    contentDiv.innerHTML = processedContent;
    
    messageDiv.appendChild(contentDiv);
    chatContainer.appendChild(messageDiv);
    
    // Scroll to bottom
    chatContainer.scrollTop = chatContainer.scrollHeight;
    
    // Apply syntax highlighting
    messageDiv.querySelectorAll('pre code').forEach((block) => {
        Prism.highlightElement(block);
    });
}

// Add thinking indicator
function addThinkingIndicator() {
    const chatContainer = document.getElementById('chatContainer');
    const thinkingDiv = document.createElement('div');
    thinkingDiv.className = 'message assistant-message';
    thinkingDiv.id = 'thinkingIndicator';
    
    thinkingDiv.innerHTML = `
        <div class="thinking">
            <span>Thinking</span>
            <div class="dots">
                <div class="dot"></div>
                <div class="dot"></div>
                <div class="dot"></div>
            </div>
        </div>
    `;
    
    chatContainer.appendChild(thinkingDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

// Remove thinking indicator
function removeThinkingIndicator() {
    const thinkingIndicator = document.getElementById('thinkingIndicator');
    if (thinkingIndicator) {
        thinkingIndicator.remove();
    }
}

// Send query to backend
async function sendQuery() {
    const queryInput = document.getElementById('queryInput');
    const queryMode = document.getElementById('queryMode');
    const query = queryInput.value.trim();
    
    if (!query) return;
    
    // Add user message
    addMessage(query, true);
    
    // Clear input and reset height
    queryInput.value = '';
    queryInput.style.height = 'auto';
    
    // Add thinking indicator
    addThinkingIndicator();
    
    try {
        const response = await fetch('http://localhost:5000/query', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                query_text: query,
                mode: queryMode.value,
                optimize: document.getElementById('optimizeToggle').checked
            })
        });
        
        const data = await response.json();
        
        if (data.status === 'error') {
            throw new Error(data.error || 'An error occurred');
        }
        
        // Remove thinking indicator
        removeThinkingIndicator();
        
        // Process response
        let responseText = data.data.text;
        
        // Extract thinking section if present
        let thinkingMatch = responseText.match(/<think>([\s\S]*?)<\/think>/);
        let thinkingText = thinkingMatch ? thinkingMatch[1].trim() : "";
        let formattedResponse = responseText.replace(/<think>[\s\S]*?<\/think>/, "").trim();
        
        // Add thinking section if present
        if (thinkingText) {
            addMessage(thinkingText);
        }
        
        // Add main response
        addMessage(formattedResponse);
        
        // Add sources if present
        if (data.data.sources && data.data.sources.length > 0) {
            const sourcesHtml = `
                <div class="sources-section">
                    <h4>Sources:</h4>
                    ${data.data.sources.map(source => `
                        <div class="source-item">${source}</div>
                    `).join('')}
                </div>
            `;
            addMessage(sourcesHtml);
        }
        
    } catch (error) {
        // Remove thinking indicator
        removeThinkingIndicator();
        
        // Show error message
        addMessage(`Error: ${error.message}`);
    }
}

// Initialize chat with welcome message
addMessage(`Welcome to the RAG Assistant! You can ask me questions about the documents in the knowledge base. Choose your preferred query mode from the sidebar:

- **RAG Mode**: Uses document context to answer questions
- **Direct Mode**: Uses the model's knowledge directly
- **Hybrid Mode**: Combines both approaches
- **Graph Mode**: Uses graph-based reasoning to answer questions
- **Light RAG Mode**: Uses a lightweight RAG implementation for faster responses

How can I help you today?`); 