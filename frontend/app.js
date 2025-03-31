// Initialize markdown-it
const md = window.markdownit({
    html: true,
    linkify: true,
    typographer: true,
    highlight: function (str, lang) {
        if (lang && Prism.languages[lang]) {
            try {
                return `<pre class="line-numbers" data-language="${lang}"><code class="language-${lang}">${Prism.highlight(str, Prism.languages[lang], lang)}</code></pre>`;
            } catch (__) {}
        }
        return `<pre class="line-numbers" data-language="plaintext"><code class="language-plaintext">${md.utils.escapeHtml(str)}</code></pre>`;
    }
});

// Rough estimation of tokens (4 characters per token on average)
function estimateTokens(text) {
    return Math.ceil(text.length / 4);
}

// Update context length display
function updateContextLength() {
    const queryInput = document.getElementById('queryInput');
    const includeHistory = document.getElementById('includeHistory');
    const contextLengthDisplay = document.getElementById('contextLength');
    
    let totalTokens = estimateTokens(queryInput.value);
    
    if (includeHistory.checked && currentChatId) {
        const chat = chats.find(c => c.id === currentChatId);
        if (chat) {
            const historyText = chat.messages
                .map(msg => msg.content)
                .join('\n');
            totalTokens += estimateTokens(historyText);
        }
    }
    
    contextLengthDisplay.textContent = `${totalTokens} tokens`;
}

// Add event listeners for context length updates
document.getElementById('queryInput').addEventListener('input', updateContextLength);
document.getElementById('includeHistory').addEventListener('change', updateContextLength);

// Auto-resize textarea
const textarea = document.getElementById('queryInput');
textarea.addEventListener('input', function() {
    this.style.height = 'auto';
    this.style.height = Math.min(this.scrollHeight, 200) + 'px';
    updateContextLength();
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
    
    // Load messages with their saved timestamps
    chat.messages.forEach(msg => {
        addMessage(msg.content, msg.isUser, msg.messageId, msg.timestamp);
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

// Message Search
const searchInput = document.getElementById('searchInput');
let searchTimeout;

searchInput.addEventListener('input', (e) => {
    clearTimeout(searchTimeout);
    searchTimeout = setTimeout(() => {
        const searchTerm = e.target.value.toLowerCase();
        const messages = document.querySelectorAll('.message');
        
        messages.forEach(message => {
            const content = message.textContent.toLowerCase();
            const isMatch = content.includes(searchTerm);
            
            message.style.display = isMatch ? 'block' : 'none';
            
            if (isMatch && searchTerm) {
                highlightText(message, searchTerm);
            } else {
                removeHighlights(message);
            }
        });
    }, 300);
});

function highlightText(element, searchTerm) {
    const content = element.textContent;
    const regex = new RegExp(`(${searchTerm})`, 'gi');
    element.innerHTML = content.replace(regex, '<span class="highlight">$1</span>');
}

function removeHighlights(element) {
    const highlights = element.querySelectorAll('.highlight');
    highlights.forEach(highlight => {
        const text = highlight.textContent;
        highlight.replaceWith(text);
    });
}

// Code Block Copy Button
function addCopyButtons() {
    document.querySelectorAll('pre code').forEach(block => {
        const pre = block.parentElement;
        const language = pre.getAttribute('data-language');
        
        // Only add copy button for code blocks with a language specified
        if (language && language !== 'plaintext') {
            const header = document.createElement('div');
            header.className = 'code-block-header';
            
            const languageSpan = document.createElement('span');
            languageSpan.className = 'code-block-language';
            languageSpan.textContent = language;
            
            const copyButton = document.createElement('button');
            copyButton.className = 'copy-button';
            copyButton.textContent = 'Copy';
            copyButton.onclick = async () => {
                try {
                    await navigator.clipboard.writeText(block.textContent);
                    copyButton.textContent = 'Copied!';
                    setTimeout(() => {
                        copyButton.textContent = 'Copy';
                    }, 2000);
                } catch (err) {
                    console.error('Failed to copy text:', err);
                }
            };
            
            header.appendChild(languageSpan);
            header.appendChild(copyButton);
            pre.insertBefore(header, block);
        }
    });
}

// Export Chat
const exportButton = document.getElementById('exportButton');

exportButton.addEventListener('click', () => {
    if (!currentChatId) return;
    
    const chat = chats.find(c => c.id === currentChatId);
    if (!chat) return;
    
    const exportData = {
        title: chat.title,
        messages: chat.messages,
        timestamp: chat.timestamp
    };
    
    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `chat-${chat.title.toLowerCase().replace(/\s+/g, '-')}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
});

// Update addMessage function to include new features
function addMessage(content, isUser = false, messageId = null, savedTimestamp = null) {
    if (!currentChatId) return;
    
    const chatContainer = document.getElementById('chatContainer');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${isUser ? 'user-message' : 'assistant-message'}`;
    
    // Use provided messageId or generate new one
    messageDiv.dataset.messageId = messageId || Date.now().toString();
    
    // Add message header with timestamp
    const header = document.createElement('div');
    header.className = 'message-header';
    
    const timestamp = document.createElement('span');
    timestamp.className = 'message-timestamp';
    // Use saved timestamp if available, otherwise use current time
    timestamp.textContent = savedTimestamp ? new Date(savedTimestamp).toLocaleTimeString() : new Date().toLocaleTimeString();
    
    const status = document.createElement('div');
    status.className = 'message-status';
    const statusIndicator = document.createElement('span');
    statusIndicator.className = 'status-indicator';
    status.appendChild(statusIndicator);
    
    header.appendChild(timestamp);
    header.appendChild(status);
    messageDiv.appendChild(header);
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    
    // Process markdown and sanitize HTML
    const processedContent = DOMPurify.sanitize(md.render(content));
    contentDiv.innerHTML = processedContent;
    
    messageDiv.appendChild(contentDiv);
    chatContainer.appendChild(messageDiv);
    
    // Scroll to bottom
    chatContainer.scrollTop = chatContainer.scrollHeight;
    
    // Apply syntax highlighting and add copy buttons
    messageDiv.querySelectorAll('pre code').forEach((block) => {
        Prism.highlightElement(block);
    });
    addCopyButtons();

    // Only save to chat history if this is a new message (not loading from history)
    if (!messageId) {
        const chat = chats.find(c => c.id === currentChatId);
        if (chat) {
            chat.messages.push({
                content,
                isUser,
                timestamp: new Date().toISOString(),
                messageId: messageDiv.dataset.messageId
            });
            
            // Update chat title if it's the first message
            if (chat.messages.length === 1 && !isUser) {
                chat.title = content.substring(0, 30) + (content.length > 30 ? '...' : '');
            }
            
            localStorage.setItem('chats', JSON.stringify(chats));
            updateChatHistoryUI();
        }
    }
}

// Initialize the app
document.addEventListener('DOMContentLoaded', () => {
    // Clear any existing messages in the chat container
    const chatContainer = document.getElementById('chatContainer');
    chatContainer.innerHTML = '';
    
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
    
    // Add copy buttons to existing code blocks
    addCopyButtons();
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

// Toggle section visibility
function toggleSection(section) {
    section.classList.toggle('collapsed');
}

// Add click handlers for section toggles
document.addEventListener('click', function(e) {
    if (e.target.closest('.hamburger')) {
        const section = e.target.closest('.special-section');
        if (section) {
            toggleSection(section);
        }
    }
});

// Update sendQuery function to include history
async function sendQuery() {
    if (!currentChatId) {
        createNewChat();
    }
    
    const queryInput = document.getElementById('queryInput');
    const queryMode = document.getElementById('queryMode');
    const includeHistory = document.getElementById('includeHistory');
    const query = queryInput.value.trim();
    
    if (!query) return;
    
    // Add user message
    addMessage(query, true);
    
    // Clear input and reset height
    queryInput.value = '';
    queryInput.style.height = 'auto';
    updateContextLength();
    
    // Add thinking indicator
    addThinkingIndicator();
    
    try {
        let requestBody = {
            query_text: query,
            mode: queryMode.value,
            optimize: document.getElementById('optimizeToggle').checked,
            hybrid: document.getElementById('hybridToggle').checked
        };

        // Include conversation history if requested
        if (includeHistory.checked) {
            const chat = chats.find(c => c.id === currentChatId);
            if (chat) {
                requestBody.conversation_history = chat.messages
                    .map(msg => ({
                        role: msg.isUser ? 'user' : 'assistant',
                        content: msg.content
                    }));
            }
        }

        const response = await fetch('http://localhost:5000/query', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestBody),
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
        
        // If response is an error message, display it directly without processing
        if (responseText.includes('Error processing')) {
            addMessage(responseText);
            return;
        }
        
        // Extract special sections
        const specialSections = {
            'think': responseText.match(/<think>([\s\S]*?)<\/think>/),
            'reasoning': responseText.match(/<reasoning>([\s\S]*?)<\/reasoning>/),
            'step': responseText.match(/<step>([\s\S]*?)<\/step>/),
            'analysis': responseText.match(/<analysis>([\s\S]*?)<\/analysis>/),
            'explanation': responseText.match(/<explanation>([\s\S]*?)<\/explanation>/),
            'solution': responseText.match(/<solution>([\s\S]*?)<\/solution>/),
            'approach': responseText.match(/<approach>([\s\S]*?)<\/approach>/),
            'conclusion': responseText.match(/<conclusion>([\s\S]*?)<\/conclusion>/),
            'summary': responseText.match(/<summary>([\s\S]*?)<\/summary>/),
            'evaluation': responseText.match(/<evaluation>([\s\S]*?)<\/evaluation>/),
            'consideration': responseText.match(/<consideration>([\s\S]*?)<\/consideration>/),
            'implementation': responseText.match(/<implementation>([\s\S]*?)<\/implementation>/)
        };
        
        // Remove all special sections from main response
        let formattedResponse = responseText;
        Object.keys(specialSections).forEach(tag => {
            formattedResponse = formattedResponse.replace(new RegExp(`<${tag}>[\\s\\S]*?<\\/${tag}>`, 'g'), "");
        });
        formattedResponse = formattedResponse.trim();
        
        // Format special sections
        let specialSectionsHTML = Object.entries(specialSections)
            .map(([tag, match]) => {
                if (!match) return "";
                const content = match[1].trim();
                return `
                    <div class="special-section">
                        <div class="hamburger">
                            <span></span>
                            <span></span>
                            <span></span>
                        </div>
                        ${content}
                    </div>
                `;
            })
            .join("");
        
        // Calculate tokens per second
        const totalTokens = estimateTokens(responseText);
        const totalTime = data.stats.total_time;
        const tokensPerSecond = totalTime > 0 ? (totalTokens / totalTime).toFixed(1) : '0.0';
        
        // Add main response with all special sections and token speed
        addMessage(`
            ${specialSectionsHTML}
            <div class="main-response">${formattedResponse}</div>
            <div class="token-speed">${tokensPerSecond} tokens/s</div>
        `);
        
        // Add sources if present
        if (data.data.sources && data.data.sources.length > 0) {
            const uniq = [...new Set(data.data.sources)];
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
        
        // Add event listeners for hamburger menus
        const specialSectionElements = document.querySelectorAll('.special-section');
        specialSectionElements.forEach(section => {
            const hamburger = section.querySelector('.hamburger');
            if (hamburger) {
                hamburger.addEventListener('click', () => {
                    section.classList.toggle('collapsed');
                });
            }
        });
        
    } catch (error) {
        // Remove thinking indicator
        removeThinkingIndicator();
        
        // Show error message
        addMessage(`Error: ${error.message}`);
    }
} 