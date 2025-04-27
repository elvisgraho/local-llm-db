function ThinkingIndicator() {
  return e(
    Paper,
    {
      elevation: 1,
      sx: {
        p: 1.5,
        mb: 2,
        display: "flex",
        alignItems: "center",
        gap: 1,
        maxWidth: "fit-content",
        alignSelf: "flex-start",
        backgroundColor: "action.hover",
      },
    },
    e(CircularProgress, { size: 20 }),
    e(Typography, { variant: "body2", color: "text.secondary" }, "Thinking...")
  );
}

function ChatMessage({ message }) {
  const isUser = message.isUser;
  const contentRef = useRef(null);

  useEffect(() => {
    if (contentRef.current && window.md && window.DOMPurify) {
      const rawHtml = md.render(message.content);
      const cleanHtml = DOMPurify.sanitize(rawHtml);
      contentRef.current.innerHTML = cleanHtml;

      // Highlight all code blocks within the message container
      // This should also trigger plugins like line-numbers if configured correctly
      if (window.Prism) {
        Prism.highlightAllUnder(contentRef.current);
      }

      // Add copy buttons after highlighting
      if (
        window.uiUtils &&
        typeof window.uiUtils.addCopyButtons === "function"
      ) {
        // TODO: Consider passing contentRef.current to addCopyButtons if global doesn't work reliably
        window.uiUtils.addCopyButtons();
      }
    }
  }, [message.content]);

  return e(
    Paper,
    {
      elevation: 1,
      sx: {
        p: 1.5,
        mb: 2,
        maxWidth: "80%",
        alignSelf: isUser ? "flex-end" : "flex-start",
        bgcolor: isUser ? "primary.light" : "background.paper",
        color: isUser ? "primary.contrastText" : "text.primary",
        overflowWrap: "break-word",
        position: "relative",
      },
    },
    e(Box, { ref: contentRef, className: "message-content-mui" })
  );
}

function ChatArea() {
  const [messages, setMessages] = useState([]);
  const [activeChatId, setActiveChatId] = useState(null);
  const [isThinking, setIsThinking] = useState(false);
  const chatContainerRef = useRef(null);

  const updateMessages = useCallback(() => {
    if (window.chatManager) {
      const currentChat = window.chatManager.getCurrentChat();
      const newMessages = currentChat ? currentChat.messages : [];
      // Ensure a new array reference is passed to setMessages
      // even if chatManager returns the same mutated array instance.
      setMessages([...newMessages]);
    } else {
      console.error("ChatArea: chatManager not found.");
      setMessages([]);
    }
  }, []);

  useEffect(() => {
    if (window.chatManager) {
      // Initial load and listener setup
      console.log("ChatArea: Setting up chatManager listeners.");
      updateMessages(); // Initial fetch
      setActiveChatId(window.chatManager.getCurrentChatId());

      window.chatManager.onHistoryUpdate(updateMessages);
      window.chatManager.onActiveChatChange((newChatId) => {
        setActiveChatId(newChatId);
        updateMessages();
      });

      // Listen for thinking status changes via global events
      const handleThinkingStart = () => setIsThinking(true);
      const handleThinkingEnd = () => setIsThinking(false);
      document.addEventListener("thinkingStart", handleThinkingStart);
      document.addEventListener("thinkingEnd", handleThinkingEnd);

      // Cleanup listeners
      return () => {
        window.chatManager.offHistoryUpdate(updateMessages);
        // Assuming the same callback was used for offActiveChatChange registration
        window.chatManager.offActiveChatChange(updateMessages);
        document.removeEventListener("thinkingStart", handleThinkingStart);
        document.removeEventListener("thinkingEnd", handleThinkingEnd);
      };
    }
  }, [updateMessages]);

  // Scroll to bottom when messages or thinking state change
  useEffect(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop =
        chatContainerRef.current.scrollHeight;
    }
  }, [messages, isThinking]);

  return e(
    Box,
    {
      ref: chatContainerRef,
      id: "chatContainer",
      sx: {
        flexGrow: 1,
        overflowY: "auto",
        p: 2,
        display: "flex",
        flexDirection: "column",
        // bgcolor: "grey.100", // Removed to inherit theme background
      },
    },
    messages.length === 0 &&
      !isThinking &&
      e(
        Typography,
        { sx: { textAlign: "center", color: "text.secondary", mt: 4 } },
        "Send a message to start chatting..."
      ),
    messages.map((msg) =>
      e(ChatMessage, { key: msg.messageId || msg.timestamp, message: msg })
    ),
    isThinking && e(ThinkingIndicator)
  );
}

// Expose thinking control globally
window.chatAreaControl = {
  startThinking: () => document.dispatchEvent(new Event("thinkingStart")),
  stopThinking: () => document.dispatchEvent(new Event("thinkingEnd")),
};

console.log("ChatArea.js component defined.");
