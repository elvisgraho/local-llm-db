// Helper for Material Icons (Assuming Icon is defined globally or imported elsewhere)
// const Icon = ({ children, sx }) => e("span", { className: "material-icons", style: sx }, children);

function InputArea({ queryMode, optimize, hybrid, selectedDbName }) {
  // Add selectedDbName prop
  // Accept props
  const [inputValue, setInputValue] = useState("");
  const [includeHistory, setIncludeHistory] = useState(true);
  const [tokenCount, setTokenCount] = useState(0); // Keep for display estimate
  const textareaRef = useRef(null);

  // Keep token estimation for display purposes
  const estimateTokens = useCallback((text) => {
    // Simple estimation
    if (!text || typeof text !== "string") return 0;
    return Math.ceil(text.length / 4);
  }, []);

  // Update token count display (remains a rough estimate based on full history)
  const updateTokenCount = useCallback(() => {
    let currentTokens = estimateTokens(inputValue);
    if (includeHistory && window.chatManager) {
      const currentChat = window.chatManager.getCurrentChat();
      if (currentChat) {
        // Estimate based on full history for display
        const historyText = currentChat.messages
          .map((msg) => msg.content)
          .join("\n");
        currentTokens += estimateTokens(historyText);
      }
    }
    setTokenCount(currentTokens);
  }, [inputValue, includeHistory, estimateTokens]);

  // Update token count effect
  useEffect(() => {
    updateTokenCount();
  }, [inputValue, includeHistory, updateTokenCount]);

  // Initial token count calculation effect
  useEffect(() => {
    updateTokenCount();
  }, [updateTokenCount]); // Run once on mount

  // Auto-resize textarea effect
  useEffect(() => {
    const textarea = textareaRef.current;
    if (textarea) {
      textarea.style.height = "auto";
      const scrollHeight = textarea.scrollHeight;
      const maxHeight = 200; // Max height constraint
      textarea.style.height = `${Math.min(scrollHeight, maxHeight)}px`;
    }
  }, [inputValue]);

  // Effect to listen for chat changes and update token count
  useEffect(() => {
    if (window.chatManager) {
      console.log(
        "InputArea: Setting up chatManager listeners for token count."
      );
      // Define the listener function (which is updateTokenCount)
      const listener = updateTokenCount;

      // Register listeners
      window.chatManager.onHistoryUpdate(listener);
      window.chatManager.onActiveChatChange(listener); // Update on chat switch too

      // Cleanup function
      return () => {
        console.log(
          "InputArea: Cleaning up chatManager listeners for token count."
        );
        window.chatManager.offHistoryUpdate(listener);
        window.chatManager.offActiveChatChange(listener);
      };
    } else {
      console.warn(
        "InputArea: chatManager not found for token count listeners."
      );
    }
  }, [updateTokenCount]); // Depend on updateTokenCount to get the latest version

  const handleInputChange = (event) => {
    setInputValue(event.target.value);
  };

  const handleHistoryToggle = (event) => {
    setIncludeHistory(event.target.checked);
  };

  // Reverted handleSend to pass original arguments
  const handleSend = () => {
    if (!inputValue.trim()) return;

    if (window.sendQuery && typeof window.sendQuery === "function") {
      // Pass query text and settings state directly
      window.sendQuery(
        inputValue,
        includeHistory,
        queryMode,
        optimize,
        hybrid,
        selectedDbName // Pass selected DB name
      );
      setInputValue(""); // Clear input after sending
    } else {
      console.error("InputArea: sendQuery function not found.");
      alert("Error: Could not send message.");
    }
  };

  const handleKeyDown = (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      handleSend();
    }
  };

  return e(
    Paper,
    {
      elevation: 2,
      sx: {
        p: 1.5,
        borderTop: 1,
        borderColor: "divider",
        bgcolor: "background.default",
      },
    },
    e(
      Box,
      { sx: { display: "flex", alignItems: "flex-end", gap: 1 } },
      e(TextField, {
        id: "queryInput",
        inputRef: textareaRef,
        placeholder: "Type your message here...",
        multiline: true,
        maxRows: 8,
        value: inputValue,
        onChange: handleInputChange,
        onKeyDown: handleKeyDown,
        variant: "outlined",
        size: "small",
        fullWidth: true,
        sx: {
          bgcolor: "background.paper",
          "& .MuiOutlinedInput-root": {
            paddingRight: "10px",
          },
          "& textarea": {
            overflowY: "auto !important",
          },
        },
      }),
      e(
        Tooltip,
        { title: "Send Message" },
        e(
          "span",
          null,
          e(
            IconButton,
            {
              color: "primary",
              onClick: handleSend,
              disabled: !inputValue.trim(),
              size: "medium",
            },
            e(Icon, null, "send")
          )
        )
      )
    ),
    e(
      Box,
      {
        sx: {
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          mt: 1,
        },
      },
      e(
        Typography,
        {
          id: "contextLength", // ID remains, but value is a rough estimate
          variant: "caption",
          color: "text.secondary",
          sx: {
            bgcolor: "action.hover",
            px: 1,
            py: 0.5,
            borderRadius: 1,
          },
        },
        `${tokenCount} tokens (estimated)` // Clarify it's an estimate
      ),
      e(FormControlLabel, {
        control: e(Checkbox, {
          id: "includeHistory",
          checked: includeHistory,
          onChange: handleHistoryToggle,
          size: "small",
          sx: { py: 0 },
        }),
        label: e(
          Typography,
          { variant: "caption", color: "text.secondary" },
          "Include conversation history"
        ),
        sx: { mr: 0 },
      })
    )
  );
}

console.log("InputArea.js component defined.");
