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
        
        // Extract thinking and reasoning sections
        let thinkingMatch = responseText.match(/<think>([\s\S]*?)<\/think>/);
        let reasoningMatch = responseText.match(/<reasoning>([\s\S]*?)<\/reasoning>/);
        let stepMatch = responseText.match(/<step>([\s\S]*?)<\/step>/);
        let analysisMatch = responseText.match(/<analysis>([\s\S]*?)<\/analysis>/);
        let explanationMatch = responseText.match(/<explanation>([\s\S]*?)<\/explanation>/);
        let solutionMatch = responseText.match(/<solution>([\s\S]*?)<\/solution>/);
        let approachMatch = responseText.match(/<approach>([\s\S]*?)<\/approach>/);
        let conclusionMatch = responseText.match(/<conclusion>([\s\S]*?)<\/conclusion>/);
        let summaryMatch = responseText.match(/<summary>([\s\S]*?)<\/summary>/);
        let evaluationMatch = responseText.match(/<evaluation>([\s\S]*?)<\/evaluation>/);
        let considerationMatch = responseText.match(/<consideration>([\s\S]*?)<\/consideration>/);
        let implementationMatch = responseText.match(/<implementation>([\s\S]*?)<\/implementation>/);
        
        let thinkingText = thinkingMatch ? thinkingMatch[1].trim() : "";
        let reasoningText = reasoningMatch ? reasoningMatch[1].trim() : "";
        let stepText = stepMatch ? stepMatch[1].trim() : "";
        let analysisText = analysisMatch ? analysisMatch[1].trim() : "";
        let explanationText = explanationMatch ? explanationMatch[1].trim() : "";
        let solutionText = solutionMatch ? solutionMatch[1].trim() : "";
        let approachText = approachMatch ? approachMatch[1].trim() : "";
        let conclusionText = conclusionMatch ? conclusionMatch[1].trim() : "";
        let summaryText = summaryMatch ? summaryMatch[1].trim() : "";
        let evaluationText = evaluationMatch ? evaluationMatch[1].trim() : "";
        let considerationText = considerationMatch ? considerationMatch[1].trim() : "";
        let implementationText = implementationMatch ? implementationMatch[1].trim() : "";
        
        // Remove all special sections from main response
        let formattedResponse = responseText
            .replace(/<think>[\s\S]*?<\/think>/g, "")
            .replace(/<reasoning>[\s\S]*?<\/reasoning>/g, "")
            .replace(/<step>[\s\S]*?<\/step>/g, "")
            .replace(/<analysis>[\s\S]*?<\/analysis>/g, "")
            .replace(/<explanation>[\s\S]*?<\/explanation>/g, "")
            .replace(/<solution>[\s\S]*?<\/solution>/g, "")
            .replace(/<approach>[\s\S]*?<\/approach>/g, "")
            .replace(/<conclusion>[\s\S]*?<\/conclusion>/g, "")
            .replace(/<summary>[\s\S]*?<\/summary>/g, "")
            .replace(/<evaluation>[\s\S]*?<\/evaluation>/g, "")
            .replace(/<consideration>[\s\S]*?<\/consideration>/g, "")
            .replace(/<implementation>[\s\S]*?<\/implementation>/g, "")
            .trim();
        
        // Format special sections
        let thinkingHTML = thinkingText ? 
            `<div class="thinking-section">
                <div class="hamburger">
                    <span></span>
                    <span></span>
                    <span></span>
                </div>
                ${thinkingText}
            </div>` : "";
        let reasoningHTML = reasoningText ? 
            `<div class="reasoning-section">
                <div class="hamburger">
                    <span></span>
                    <span></span>
                    <span></span>
                </div>
                ${reasoningText}
            </div>` : "";
        let stepHTML = stepText ? 
            `<div class="step-section">
                <div class="hamburger">
                    <span></span>
                    <span></span>
                    <span></span>
                </div>
                ${stepText}
            </div>` : "";
        let analysisHTML = analysisText ? 
            `<div class="analysis-section">
                <div class="hamburger">
                    <span></span>
                    <span></span>
                    <span></span>
                </div>
                ${analysisText}
            </div>` : "";
        let explanationHTML = explanationText ? 
            `<div class="explanation-section">
                <div class="hamburger">
                    <span></span>
                    <span></span>
                    <span></span>
                </div>
                ${explanationText}
            </div>` : "";
        let solutionHTML = solutionText ? 
            `<div class="solution-section">
                <div class="hamburger">
                    <span></span>
                    <span></span>
                    <span></span>
                </div>
                ${solutionText}
            </div>` : "";
        let approachHTML = approachText ? 
            `<div class="approach-section">
                <div class="hamburger">
                    <span></span>
                    <span></span>
                    <span></span>
                </div>
                ${approachText}
            </div>` : "";
        let conclusionHTML = conclusionText ? 
            `<div class="conclusion-section">
                <div class="hamburger">
                    <span></span>
                    <span></span>
                    <span></span>
                </div>
                ${conclusionText}
            </div>` : "";
        let summaryHTML = summaryText ? 
            `<div class="summary-section">
                <div class="hamburger">
                    <span></span>
                    <span></span>
                    <span></span>
                </div>
                ${summaryText}
            </div>` : "";
        let evaluationHTML = evaluationText ? 
            `<div class="evaluation-section">
                <div class="hamburger">
                    <span></span>
                    <span></span>
                    <span></span>
                </div>
                ${evaluationText}
            </div>` : "";
        let considerationHTML = considerationText ? 
            `<div class="consideration-section">
                <div class="hamburger">
                    <span></span>
                    <span></span>
                    <span></span>
                </div>
                ${considerationText}
            </div>` : "";
        let implementationHTML = implementationText ? 
            `<div class="implementation-section">
                <div class="hamburger">
                    <span></span>
                    <span></span>
                    <span></span>
                </div>
                ${implementationText}
            </div>` : "";
        
        // Add main response with all special sections
        addMessage(`
            ${thinkingHTML}
            ${reasoningHTML}
            ${stepHTML}
            ${analysisHTML}
            ${explanationHTML}
            ${solutionHTML}
            ${approachHTML}
            ${conclusionHTML}
            ${summaryHTML}
            ${evaluationHTML}
            ${considerationHTML}
            ${implementationHTML}
            <div class="main-response">${formattedResponse}</div>
        `);
        
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
        
        // Add event listeners for hamburger menus
        const sections = document.querySelectorAll('.thinking-section, .reasoning-section, .step-section, .analysis-section, .explanation-section, .solution-section, .approach-section, .conclusion-section, .summary-section, .evaluation-section, .consideration-section, .implementation-section');
        sections.forEach(section => {
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