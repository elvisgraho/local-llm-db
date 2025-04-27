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
    const chatList = getChatList();
    historyListeners.forEach((listener) => listener(chatList));
  }

  function _notifyActiveChatListeners() {
    activeChatListeners.forEach((listener) => listener(currentChatId));
  }

  function _generateChatId() {
    return `chat_${Date.now()}_${Math.random().toString(36).substring(2, 7)}`;
  }

  function _updateChatTitleIfNeeded(chat, newMessageData) {
    if (
      chat &&
      chat.title === "New Chat" &&
      chat.messages.length > 0 &&
      !newMessageData.isUser
    ) {
      const firstAssistantMessage = chat.messages.find((m) => !m.isUser);
      if (firstAssistantMessage) {
        const content = firstAssistantMessage.content || "";
        const firstWords = content.split(" ").slice(0, 5).join(" ");
        chat.title =
          firstWords + (content.length > firstWords.length ? "..." : "");
        console.log(`Chat title updated to: "${chat.title}"`);
        return true;
      }
    }
    return false;
  }

  // --- Public API ---

  function getChatList() {
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

    currentChatId = chatId;
    _notifyHistoryListeners();
    _notifyActiveChatListeners();
    return chatId;
  }

  function switchToChat(chatId) {
    if (chatId === currentChatId) return;

    const chatExists = chats.some((c) => c.id === chatId);
    if (!chatExists) {
      console.error("Cannot switch to non-existent chat:", chatId);
      if (chats.length > 0) {
        const sortedList = getChatList();
        switchToChat(sortedList[0].id);
      } else {
        createNewChat();
      }
      return;
    }

    currentChatId = chatId;
    console.log("Switched to chat:", currentChatId);
    _notifyActiveChatListeners();
  }

  function deleteChat(chatId) {
    const chatIndex = chats.findIndex((chat) => chat.id === chatId);
    if (chatIndex === -1) {
      console.warn("Cannot delete non-existent chat:", chatId);
      return;
    }

    const deletedChatTitle = chats[chatIndex].title;
    chats.splice(chatIndex, 1);
    _saveChats();
    console.log(`Deleted chat "${deletedChatTitle}" (${chatId})`);

    let nextChatId = null;
    if (currentChatId === chatId) {
      if (chats.length > 0) {
        const sortedChats = getChatList();
        nextChatId = sortedChats[0].id;
      } else {
        nextChatId = createNewChat();
        return;
      }
    }

    _notifyHistoryListeners();

    if (nextChatId && nextChatId !== currentChatId) {
      switchToChat(nextChatId);
    }
  }

  function addMessageToHistory(messageData) {
    if (!currentChatId) {
      console.error("Cannot add message: No current chat selected.");
      return;
    }
    const chat = getCurrentChat();
    if (chat) {
      if (!messageData.messageId) {
        messageData.messageId = `msg_${Date.now()}_${Math.random()
          .toString(36)
          .substring(2, 7)}`;
      }
      chat.messages.push(messageData);
      chat.timestamp = new Date().toISOString();

      _updateChatTitleIfNeeded(chat, messageData);
      _saveChats();
      console.log(
        "Message added to chat:",
        currentChatId,
        messageData.messageId
      );
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
      createNewChat();
    } else {
      const sortedList = getChatList();
      currentChatId = sortedList[0].id;
      console.log("ChatManager initialized. Current chat:", currentChatId);
    }
  }

  initialize();

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
