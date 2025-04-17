// frontend/js/chatManager.js - Refactored State Manager

(function () {
  // IIFE to encapsulate state and avoid polluting global scope further
  // --- State ---
  let chats = [];
  let currentChatId = null;
  let historyListeners = [];
  let activeChatListeners = [];

  // --- Private Helper Functions ---

  function _saveChats() {
    try {
      localStorage.setItem("chats", JSON.stringify(chats));
      console.log("Chats saved to localStorage.");
    } catch (error) {
      console.error("Error saving chats to localStorage:", error);
      // Consider notifying the user or implementing a fallback
    }
  }

  function _loadChats() {
    try {
      const savedChats = localStorage.getItem("chats");
      chats = savedChats ? JSON.parse(savedChats) : [];
      console.log("Chats loaded from localStorage.");
    } catch (error) {
      console.error("Error loading chats from localStorage:", error);
      chats = []; // Reset to empty array on error
    }
  }

  function _notifyHistoryListeners() {
    console.log("Notifying history listeners...");
    // Provide a copy of the chat list to listeners
    const chatList = getChatList();
    historyListeners.forEach((listener) => listener(chatList));
  }

  function _notifyActiveChatListeners() {
    console.log(`Notifying active chat listeners: ${currentChatId}`);
    activeChatListeners.forEach((listener) => listener(currentChatId));
  }

  function _generateChatId() {
    return `chat_${Date.now()}_${Math.random().toString(36).substring(2, 7)}`;
  }

  function _updateChatTitleIfNeeded(chat, newMessageData) {
    if (
      chat &&
      chat.title === "New Chat" &&
      chat.messages.length > 0 && // Ensure message was actually added
      !newMessageData.isUser // Only update on first assistant message
    ) {
      // Find the first assistant message to base the title on
      const firstAssistantMessage = chat.messages.find((m) => !m.isUser);
      if (firstAssistantMessage) {
        const content = firstAssistantMessage.content || "";
        const firstWords = content.split(" ").slice(0, 5).join(" ");
        chat.title =
          firstWords + (content.length > firstWords.length ? "..." : "");
        console.log(`Chat title updated to: "${chat.title}"`);
        return true; // Indicate title was updated
      }
    }
    return false; // Title not updated
  }

  // --- Public API ---

  function getChatList() {
    // Return only ID and title, sorted by timestamp descending
    return [...chats]
      .sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp))
      .map((chat) => ({ id: chat.id, title: chat.title }));
  }

  function getCurrentChatId() {
    return currentChatId;
  }

  function getCurrentChat() {
    if (!currentChatId) return null;
    return chats.find((chat) => chat.id === currentChatId);
  }

  function createNewChat() {
    const chatId = _generateChatId();
    const newChat = {
      id: chatId,
      title: "New Chat",
      messages: [],
      timestamp: new Date().toISOString(),
    };

    chats.push(newChat);
    _saveChats();
    console.log("New chat created:", chatId);

    // Set as current and notify
    currentChatId = chatId;
    _notifyHistoryListeners(); // New chat added to list
    _notifyActiveChatListeners(); // Active chat changed
    return chatId;
  }

  function switchToChat(chatId) {
    if (chatId === currentChatId) return; // No change

    const chatExists = chats.some((c) => c.id === chatId);
    if (!chatExists) {
      console.error("Cannot switch to non-existent chat:", chatId);
      // Optionally switch to the first available chat or create new
      if (chats.length > 0) {
        switchToChat(chats[0].id); // Switch to the first one (after sorting)
      } else {
        createNewChat();
      }
      return;
    }

    currentChatId = chatId;
    console.log("Switched to chat:", currentChatId);
    _notifyActiveChatListeners(); // Notify components about the change
    // History listeners are not notified as the list itself didn't change
  }

  function deleteChat(chatId) {
    const chatIndex = chats.findIndex((chat) => chat.id === chatId);
    if (chatIndex === -1) {
      console.warn("Cannot delete non-existent chat:", chatId);
      return;
    }

    const deletedChatTitle = chats[chatIndex].title;
    chats.splice(chatIndex, 1); // Remove chat from array
    _saveChats();
    console.log(`Deleted chat "${deletedChatTitle}" (${chatId})`);

    let nextChatId = null;
    if (currentChatId === chatId) {
      // If deleted chat was current, switch to most recent remaining chat
      if (chats.length > 0) {
        const sortedChats = getChatList(); // Gets sorted list
        nextChatId = sortedChats[0].id;
      } else {
        // No chats left, create a new one
        nextChatId = createNewChat(); // This will set currentChatId and notify
        return; // createNewChat already notified
      }
    }

    // Notify history listeners first (list has changed)
    _notifyHistoryListeners();

    // If we needed to switch chats, do it now and notify active listeners
    if (nextChatId && nextChatId !== currentChatId) {
      switchToChat(nextChatId); // This will notify active listeners
    }
  }

  function addMessageToHistory(messageData) {
    if (!currentChatId) {
      console.error("Cannot add message: No current chat selected.");
      return;
    }
    const chat = getCurrentChat();
    if (chat) {
      // Ensure message has a unique ID if not provided
      if (!messageData.messageId) {
        messageData.messageId = `msg_${Date.now()}_${Math.random()
          .toString(36)
          .substring(2, 7)}`;
      }
      chat.messages.push(messageData);
      chat.timestamp = new Date().toISOString(); // Update chat timestamp on new message

      const titleUpdated = _updateChatTitleIfNeeded(chat, messageData);

      _saveChats();
      console.log(
        "Message added to chat:",
        currentChatId,
        messageData.messageId
      );

      // Notify history listeners (content changed, maybe title/order changed)
      _notifyHistoryListeners();
    } else {
      console.error("Cannot add message: Current chat object not found.");
    }
  }

  // --- Listener Management ---

  function onHistoryUpdate(callback) {
    if (typeof callback === "function") {
      historyListeners.push(callback);
    }
  }

  function offHistoryUpdate(callback) {
    historyListeners = historyListeners.filter(
      (listener) => listener !== callback
    );
  }

  function onActiveChatChange(callback) {
    if (typeof callback === "function") {
      activeChatListeners.push(callback);
    }
  }

  function offActiveChatChange(callback) {
    activeChatListeners = activeChatListeners.filter(
      (listener) => listener !== callback
    );
  }

  // --- Initialization ---
  function initialize() {
    _loadChats();
    if (chats.length === 0) {
      createNewChat(); // Creates first chat and sets it as current
    } else {
      // Set the most recent chat as current initially, but don't trigger full switch logic
      const sortedList = getChatList(); // Gets sorted list
      currentChatId = sortedList[0].id;
      console.log("ChatManager initialized. Current chat:", currentChatId);
      // Components will read initial state on mount
    }
  }

  initialize(); // Initialize on script load

  // --- Global Exposure ---
  window.chatManager = {
    // Data Access
    getChatList,
    getCurrentChatId,
    getCurrentChat,
    // Actions
    createNewChat,
    switchToChat,
    deleteChat,
    addMessageToHistory,
    // Listener Registration
    onHistoryUpdate,
    offHistoryUpdate,
    onActiveChatChange,
    offActiveChatChange,
  };

  console.log("chatManager.js refactored and initialized.");
})(); // End IIFE
