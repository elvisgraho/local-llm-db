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

// Chat management
let chats = JSON.parse(localStorage.getItem('chats')) || [];
let currentChatId = null;

// Create a new chat
function createNewChat() {
    const chatId = Date.now().toString();
    const newChat = {
        id: chatId,
        title: 'New Chat',
        messages: [],
        timestamp: new Date().toISOString()
    };
    
    chats.push(newChat);
    localStorage.setItem('chats', JSON.stringify(chats));
    
    // Switch to the new chat
    switchChat(chatId);
}

// Switch to a specific chat
function switchChat(chatId) {
    currentChatId = chatId;
    const chat = chats.find(c => c.id === chatId);
    if (!chat) return;
    
    // Clear current chat container
    const chatContainer = document.getElementById('chatContainer');
    chatContainer.innerHTML = '';
    
    // Load messages
    chat.messages.forEach(msg => {
        addMessage(msg.content, msg.isUser);
    });
    
    // Update chat history UI
    updateChatHistoryUI();
}

// Delete a chat
function deleteChat(chatId, event) {
    event.stopPropagation(); // Prevent chat switching when clicking delete
    
    // Remove chat from array
    chats = chats.filter(chat => chat.id !== chatId);
    
    // If deleted chat was current, switch to most recent chat or create new one
    if (currentChatId === chatId) {
        if (chats.length > 0) {
            const mostRecentChat = chats.reduce((latest, current) => 
                new Date(current.timestamp) > new Date(latest.timestamp) ? current : latest
            );
            switchChat(mostRecentChat.id);
        } else {
            createNewChat();
        }
    }
    
    // Update localStorage and UI
    localStorage.setItem('chats', JSON.stringify(chats));
    updateChatHistoryUI();
}

// Update chat history UI
function updateChatHistoryUI() {
    const chatHistoryList = document.getElementById('chatHistoryList');
    chatHistoryList.innerHTML = '';
    
    chats.forEach(chat => {
        const chatItem = document.createElement('div');
        chatItem.className = `chat-history-item ${chat.id === currentChatId ? 'active' : ''}`;
        
        chatItem.onclick = () => switchChat(chat.id);


        const titleSpan = document.createElement('span');
        titleSpan.textContent = chat.title;
        
        
        const deleteBtn = document.createElement('button');
        deleteBtn.className = 'delete-btn';
        deleteBtn.innerHTML = '×';
        deleteBtn.onclick = (e) => deleteChat(chat.id, e);
        
        chatItem.appendChild(titleSpan);
        chatItem.appendChild(deleteBtn);
        chatHistoryList.appendChild(chatItem);
    });
}

// Add message to chat
function addMessage(content, isUser = false) {
    if (!currentChatId) return;
    
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

    // Save to chat history
    const chat = chats.find(c => c.id === currentChatId);
    if (chat) {
        chat.messages.push({
            content,
            isUser,
            timestamp: new Date().toISOString()
        });
        
        // Update chat title if it's the first message
        if (chat.messages.length === 1 && !isUser) {
            chat.title = content.substring(0, 30) + (content.length > 30 ? '...' : '');
        }
        
        localStorage.setItem('chats', JSON.stringify(chats));
        updateChatHistoryUI();
    }
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
    if (!currentChatId) {
        createNewChat();
    }
    
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
                optimize: document.getElementById('optimizeToggle').checked,
                hybrid: document.getElementById('hybridToggle').checked
            }),
            // Add timeout of 5 minutes (300000ms)
            signal: AbortSignal.timeout(300000)
        });
        
        const data = await response.json();
        
        if (data.status === 'error') {
            throw new Error(data.error || 'An error occurred');
        }
        
        // Remove thinking indicator
        removeThinkingIndicator();
        
        // Process response
        let responseText = data.data.text;
        
        // Remove all thinking sections
        let formattedResponse = responseText.replace(/<think>[\s\S]*?<\/think>/g, "").trim();
        
        // Add main response
        addMessage(formattedResponse);
        
        // Add sources if present
        if (data.data.sources && data.data.sources.length > 0) {
            uniq = [...new Set(data.data.sources)];
            const sourcesList = document.createElement('ul');
            sourcesList.className = 'sources-list';
            
            uniq.forEach(source => {
                if (typeof(source) === 'string') {
                    const filename = source.split('\\').pop();
                    const listItem = document.createElement('li');
                    const link = document.createElement('a');
                    link.href = source;
                    link.textContent = filename;
                    link.onclick = (e) => {
                        e.preventDefault();
                        openLocalFile(source);
                    };
                    listItem.appendChild(link);
                    sourcesList.appendChild(listItem);
                }
            });
            
            const sourcesDiv = document.createElement('div');
            sourcesDiv.className = 'message assistant-message';
            sourcesDiv.innerHTML = '<strong>Sources:</strong>';
            sourcesDiv.appendChild(sourcesList);
            document.getElementById('chatContainer').appendChild(sourcesDiv);
        }
        
    } catch (error) {
        // Remove thinking indicator
        removeThinkingIndicator();
        
        // Show error message
        addMessage(`Error: ${error.message}`);
    }
}

// Initialize the app
document.addEventListener('DOMContentLoaded', () => {
    // Create initial chat if none exists
    if (chats.length === 0) {
        createNewChat();
    } else {
        // Switch to the most recent chat
        const mostRecentChat = chats.reduce((latest, current) => 
            new Date(current.timestamp) > new Date(latest.timestamp) ? current : latest
        );
        switchChat(mostRecentChat.id);
    }
});

// Add this function at the end of the file, before the DOMContentLoaded event listener
async function openLocalFile(filePath) {
    try {
        const response = await fetch('http://localhost:5000/open_file', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ file_path: filePath })
        });
        
        const data = await response.json();
        if (data.status === 'error') {
            throw new Error(data.error || 'Failed to open file');
        }
    } catch (error) {
        console.error('Error opening file:', error);
        alert('Failed to open file. Please check if the file exists and you have permission to access it.');
    }
} 