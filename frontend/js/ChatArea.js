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
      let cleanHtml = DOMPurify.sanitize(rawHtml);

      // --- Add clickable source links ---
      const sourcePattern = /\[Source: (.*?)\]/g;
      const finalHtml = cleanHtml.replace(sourcePattern, (match, filePath) => {
        try {
          // Trim potential whitespace
          const trimmedPath = filePath.trim();

          // --- Create VSCode URI ---
          let uriPath = trimmedPath.replace(/\\/g, "/");
          // Ensure it starts with a slash if it's a drive path like C:/...
          if (/^[a-zA-Z]:\//.test(uriPath)) {
            uriPath = "/" + uriPath;
          }
          // Encode for URI, preserving slashes and drive colon.
          // We split by slash, encode each part EXCEPT the drive letter part if present.
          const encodedPath = uriPath
            .split("/")
            .map((part, index) => {
              // Check if it's the drive letter part (e.g., '/C:')
              if (index === 1 && /^[a-zA-Z]:$/.test(part)) {
                return part; // Don't encode drive letter
              }
              // Encode other parts, including filenames with spaces, #, etc.
              return encodeURIComponent(part);
            })
            .join("/");
          const vscodeUri = `vscode://file${encodedPath}`;
          // --- End VSCode URI ---

          // Extract filename
          const filename = trimmedPath.split(/[\\/]/).pop() || trimmedPath; // Get part after last slash/backslash

          console.log(
            `Creating VS Code link for path: ${trimmedPath} -> ${vscodeUri}`
          ); // Debugging

          // Return styled link with icon and filename
          // Using inline styles for a self-contained example. Consider moving to CSS classes.
          // Added vertical-align: middle to align better with surrounding text.
          // Added small padding and background for a "chip" look.
          return (
            `<a href="${vscodeUri}" title="Open ${trimmedPath} in VS Code" style="text-decoration: none; color: inherit; display: inline-flex; align-items: center; vertical-align: middle; margin: 0 2px; padding: 1px 5px; background-color: rgba(128, 128, 128, 0.1); border-radius: 4px; font-size: 0.9em; white-space: nowrap;">` +
            // Use standard 'material-icons' class to match index.html
            `<span class="material-icons" style="font-size: 1.1em; margin-right: 4px; opacity: 0.7; vertical-align: middle;">description</span>` + // Material file icon
            `<span style="text-decoration: underline; vertical-align: middle;">${filename}</span>` + // Filename only, underlined
            `</a>`
          );
        } catch (e) {
          console.error(
            "Error creating VS Code file link:",
            e,
            "for path:",
            filePath
          );
          return match; // Return original text if encoding/processing fails
        }
      });
      contentRef.current.innerHTML = finalHtml; // Use the HTML with links
      // --- End source link modification ---

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
    e(Box, { ref: contentRef, className: "message-content-mui" }),
    // Add source rendering logic here
    message.sources &&
      message.sources.length > 0 &&
      e(
        Box,
        {
          sx: {
            mt: 1, // Margin top to separate from main content
            pt: 1, // Padding top
            borderTop: 1, // Separator line
            borderColor: "divider",
            fontSize: "0.8rem", // Smaller font size
            color: isUser ? "primary.contrastText" : "text.secondary", // Adjust color based on sender
            opacity: 0.8, // Slightly faded
          },
        },
        e(
          Typography,
          {
            variant: "caption",
            component: "strong",
            sx: { display: "block", mb: 0.5 },
          },
          "Sources:"
        ),
        e(
          "ul",
          { style: { paddingLeft: "20px", margin: 0 } },
          message.sources.map((source, index) =>
            e(
              "li",
              { key: index },
              // Render source path with a copy button
              typeof source === "string"
                ? e(
                    Box,
                    { sx: { display: "flex", alignItems: "center", gap: 0.5 } },
                    e("span", { style: { wordBreak: "break-all" } }, source), // Display path
                    e(
                      Tooltip,
                      { title: "Copy path" },
                      e(
                        IconButton,
                        {
                          size: "small",
                          onClick: () => {
                            navigator.clipboard
                              .writeText(source)
                              .then(() => console.log("Path copied:", source)) // Optional: Add visual feedback later
                              .catch((err) =>
                                console.error("Failed to copy path:", err)
                              );
                          },
                          sx: {
                            padding: "2px", // Smaller padding
                            color: "inherit", // Inherit color
                            opacity: 0.7, // Slightly faded
                            "&:hover": { opacity: 1 },
                          },
                        },
                        e(Icon, { sx: { fontSize: "1rem" } }, "content_copy") // Smaller icon
                      )
                    )
                  )
                : JSON.stringify(source) // Fallback for non-string sources
            )
          )
        )
      )
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
