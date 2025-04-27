// Helper for Material Icons (Assuming Icon is defined globally or imported elsewhere)
// const Icon = ({ children, sx }) => e("span", { className: "material-icons", style: sx }, children);

function InputArea({ queryMode, optimize, hybrid }) {
  // Accept props
  const [inputValue, setInputValue] = useState("");
  const [includeHistory, setIncludeHistory] = useState(true);
  const [tokenCount, setTokenCount] = useState(0);
  const textareaRef = useRef(null);

  const estimateTokens = useCallback((text) => {
    // Simple estimation
    return Math.ceil((text || "").length / 4);
  }, []);

  const updateTokenCount = useCallback(() => {
    let currentTokens = estimateTokens(inputValue);
    if (includeHistory && window.chatManager) {
      const currentChat = window.chatManager.getCurrentChat();
      if (currentChat) {
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

  const handleInputChange = (event) => {
    setInputValue(event.target.value);
  };

  const handleHistoryToggle = (event) => {
    setIncludeHistory(event.target.checked);
  };

  const handleSend = () => {
    if (!inputValue.trim()) return;

    // Remove code updating hidden inputs - no longer needed

    if (window.sendQuery && typeof window.sendQuery === "function") {
      // Pass query text and settings state to sendQuery
      window.sendQuery(inputValue, includeHistory, queryMode, optimize, hybrid);
      setInputValue("");
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
        // Wrap IconButton in a span to allow Tooltip events when disabled
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
            e(Icon, null, "send") // Assumes Icon component is available globally
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
          id: "contextLength",
          variant: "caption",
          color: "text.secondary",
          sx: {
            bgcolor: "action.hover",
            px: 1,
            py: 0.5,
            borderRadius: 1,
          },
        },
        `${tokenCount} tokens`
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
