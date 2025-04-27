// Relies on global React hooks (useState, useEffect, useCallback, createElement as e)
// Relies on global MaterialUI components (Drawer, Box, List, ListItem, ListItemText, Divider, Typography, TextField, Select, MenuItem, FormControl, InputLabel, Switch, FormControlLabel, Button, IconButton, Tooltip)
// Relies on global Icon helper
// All defined in app.js

const SIDEBAR_WIDTH = 280;

function Sidebar({
  queryMode,
  optimize, // Renamed from optimizeResponse to match prop
  hybrid, // Renamed from hybridMode to match prop
  onQueryModeChange,
  onOptimizeChange,
  onHybridChange,
  selectedDbName, // Added prop
  onDbNameChange, // Added prop
}) {
  // Use the globally defined hooks and components
  // State for queryMode, optimize, hybrid is now managed by AppRoot
  const [searchTerm, setSearchTerm] = useState("");
  const [chatHistory, setChatHistory] = useState([]);
  const [activeChatId, setActiveChatId] = useState(null);
  const [availableDbs, setAvailableDbs] = useState([]); // State for DB list
  const [dbLoadingError, setDbLoadingError] = useState(null); // State for DB loading errors

  // Load settings and history on mount
  useEffect(() => {
    // Load settings logic removed - handled by AppRoot

    if (window.chatManager) {
      updateChatHistoryList();
      setActiveChatId(window.chatManager.getCurrentChatId());

      window.chatManager.onHistoryUpdate(updateChatHistoryList);
      window.chatManager.onActiveChatChange(setActiveChatId);

      return () => {
        window.chatManager.offHistoryUpdate(updateChatHistoryList);
        window.chatManager.offActiveChatChange(setActiveChatId);
      };
    } else {
      console.error("Sidebar: chatManager not found on mount.");
    }
  }, []); // Empty dependency array ensures this runs only once on mount

  // Save settings logic removed - handled by AppRoot

  // Fetch available databases when queryMode changes
  useEffect(() => {
    const fetchDbs = async () => {
      if (["rag", "kag", "lightrag"].includes(queryMode)) {
        setDbLoadingError(null); // Clear previous errors
        setAvailableDbs([]); // Clear previous list while loading
        try {
          const response = await fetch(`/api/list_dbs/${queryMode}`);
          if (!response.ok) {
            const errorData = await response.json();
            throw new Error(
              errorData.error || `HTTP error! status: ${response.status}`
            );
          }
          const data = await response.json();
          if (data.status === "success") {
            const fetchedDbs = data.databases || [];
            setAvailableDbs(fetchedDbs);

            // Auto-select if only one DB is available
            if (fetchedDbs.length === 1) {
              console.log(
                `Auto-selecting the only available DB: ${fetchedDbs[0]}`
              );
              // Only change if the current selection is different
              if (selectedDbName !== fetchedDbs[0]) {
                onDbNameChange(fetchedDbs[0]);
              }
            }
            // If current selection is not in the new list (and list isn't just one item), reset to default
            else if (selectedDbName && !fetchedDbs.includes(selectedDbName)) {
              console.log(
                `Current selection '${selectedDbName}' not in new list, resetting.`
              );
              onDbNameChange(
                window.GLOBAL_CONFIG?.DEFAULT_DB_NAME || "default"
              ); // Reset to default
            }
            // If no DB is selected but multiple are available, select the default if it exists
            else if (!selectedDbName && fetchedDbs.length > 0) {
              const defaultDb =
                window.GLOBAL_CONFIG?.DEFAULT_DB_NAME || "default";
              if (fetchedDbs.includes(defaultDb)) {
                console.log(
                  `Selecting default DB '${defaultDb}' as none was selected.`
                );
                onDbNameChange(defaultDb);
              }
              // Optional: Select the first one if default isn't present and list isn't empty?
              // else if (fetchedDbs.length > 0) { onDbNameChange(fetchedDbs[0]); }
            }
          } else {
            throw new Error(data.error || "Failed to fetch databases");
          }
        } catch (error) {
          console.error(`Error fetching databases for ${queryMode}:`, error);
          setDbLoadingError(
            `Failed to load ${queryMode} databases: ${error.message}`
          );
          setAvailableDbs([]); // Ensure list is empty on error
        }
      } else {
        setAvailableDbs([]); // Clear list if not a DB-specific mode
        setDbLoadingError(null); // Clear errors
      }
    };
    fetchDbs();
  }, [queryMode, selectedDbName, onDbNameChange]); // Re-run if mode changes or selection needs reset

  const updateChatHistoryList = useCallback(() => {
    if (window.chatManager) {
      setChatHistory(window.chatManager.getChatList());
    }
  }, []);

  // Use handlers passed via props
  const handleQueryModeChange = (event) =>
    onQueryModeChange(event.target.value);
  const handleOptimizeChange = (event) =>
    onOptimizeChange(event.target.checked);
  const handleHybridChange = (event) => onHybridChange(event.target.checked);
  // Handler for DB name selection
  const handleDbNameChange = (event) => onDbNameChange(event.target.value);

  const handleSearchChange = (event) => {
    const newSearchTerm = event.target.value;
    setSearchTerm(newSearchTerm);
    // TODO: Implement actual search filtering if needed
    console.log("Search term:", newSearchTerm);
    // Optional: Trigger global search if uiUtils handles it
    if (window.uiUtils?.handleSearchInput) {
      const searchInputEl = document.getElementById("sidebarSearchInput");
      if (searchInputEl) {
        searchInputEl.value = newSearchTerm;
        window.uiUtils.handleSearchInput();
      }
    }
  };

  const handleOpenLlmSettings = () => {
    // Prefer calling an exposed function from the modal component
    if (window.llmSettingsModalControl?.open) {
      window.llmSettingsModalControl.open();
    } else {
      // Fallback: Click the button programmatically
      const btn = document.getElementById("openLlmSettingsBtn");
      if (btn) btn.click();
      else console.warn("Cannot open LLM settings modal.");
    }
  };

  const handleNewChat = () => {
    if (window.chatManager) {
      window.chatManager.createNewChat();
    } else {
      console.error("Sidebar: chatManager not found for new chat.");
    }
  };

  const handleSelectChat = (chatId) => {
    if (window.chatManager) {
      window.chatManager.switchToChat(chatId);
    } else {
      console.error("Sidebar: chatManager not found for switching chat.");
    }
  };

  const handleDeleteChat = (event, chatId) => {
    event.stopPropagation();
    const chatTitle = chatHistory.find((c) => c.id === chatId)?.title || chatId;
    if (
      window.chatManager &&
      confirm(`Are you sure you want to delete chat "${chatTitle}"?`)
    ) {
      window.chatManager.deleteChat(chatId);
    } else if (!window.chatManager) {
      console.error("Sidebar: chatManager not found for deleting chat.");
    }
  };

  const drawerContent = e(
    Box,
    { sx: { display: "flex", flexDirection: "column", height: "100vh" } },
    // Header
    e(
      Box,
      { sx: { p: 2, borderBottom: 1, borderColor: "divider" } },
      e(
        Typography,
        { variant: "h6", component: "h1", noWrap: true },
        "RAG Assistant"
      )
    ),
    // Controls
    e(
      Box,
      { sx: { p: 2 } },
      e(TextField, {
        id: "sidebarSearchInput",
        label: "Search Chats",
        variant: "outlined",
        size: "small",
        fullWidth: true,
        value: searchTerm,
        onChange: handleSearchChange,
        sx: { mb: 2 },
      }),
      e(
        FormControl,
        { fullWidth: true, size: "small", sx: { mb: 2 } },
        e(InputLabel, { id: "query-mode-label" }, "Query Mode"),
        e(
          Select,
          {
            labelId: "query-mode-label",
            id: "queryMode",
            value: queryMode,
            label: "Query Mode",
            onChange: handleQueryModeChange,
          },
          e(
            MenuItem,
            { value: "rag" },
            e(
              Tooltip,
              {
                title:
                  "Retrieval-Augmented Generation. Best for general question answering using the knowledge base. Retrieves relevant documents and uses them to generate an answer.",
                placement: "right",
                componentsProps: {
                  tooltip: {
                    sx: {
                      bgcolor: "common.black", // Or use theme.palette.tooltip.background if available
                      "& .MuiTooltip-arrow": {
                        color: "common.black", // Match arrow color
                      },
                    },
                  },
                },
              },
              e("span", null, "RAG Mode") // Wrap text in a span for Tooltip anchor
            )
          ),
          e(MenuItem, { value: "direct" }, "Direct Mode"), // No tooltip needed
          e(
            MenuItem,
            { value: "lightrag" },
            e(
              Tooltip,
              {
                title:
                  "Faster RAG. Uses a smaller/optimized index for quicker retrieval. Good for less complex queries or when speed is prioritized over depth.",
                placement: "right",
                componentsProps: {
                  tooltip: {
                    sx: {
                      bgcolor: "common.black",
                      "& .MuiTooltip-arrow": {
                        color: "common.black",
                      },
                    },
                  },
                },
              },
              e("span", null, "Light RAG Mode")
            )
          ),
          e(
            MenuItem,
            { value: "kag" },
            e(
              Tooltip,
              {
                title:
                  "Knowledge-Augmented Generation. Focuses on structured knowledge (if available). Better for queries needing specific facts or relationships from structured data.",
                placement: "right",
                componentsProps: {
                  tooltip: {
                    sx: {
                      bgcolor: "common.black",
                      "& .MuiTooltip-arrow": {
                        color: "common.black",
                      },
                    },
                  },
                },
              },
              e("span", null, "KAG Mode")
            )
          )
        )
      ),
      // Conditionally render the DB selector based on queryMode
      ["rag", "kag", "lightrag"].includes(queryMode) &&
        e(
          FormControl,
          {
            fullWidth: true,
            size: "small",
            sx: { mb: 2 },
            error: !!dbLoadingError,
          },
          e(InputLabel, { id: "db-name-label" }, "Database Name"),
          e(
            Select,
            {
              labelId: "db-name-label",
              id: "dbName",
              value: selectedDbName || "", // Use prop, handle empty/null
              label: "Database Name",
              onChange: handleDbNameChange, // Use new handler
              disabled: availableDbs.length === 0 && !dbLoadingError, // Disable if empty and no error
            },
            // Display error message if loading failed
            dbLoadingError &&
              e(
                MenuItem,
                { disabled: true, value: "" },
                e(
                  Typography,
                  { color: "error", variant: "caption" },
                  dbLoadingError
                )
              ),
            // Display message if loading but empty
            !dbLoadingError &&
              availableDbs.length === 0 &&
              e(
                MenuItem,
                { disabled: true, value: "" },
                `No ${queryMode} DBs found`
              ),
            // Add default option if it exists or if list is empty
            // (window.GLOBAL_CONFIG.DEFAULT_DB_NAME || 'default') &&
            //   e(MenuItem, { value: window.GLOBAL_CONFIG.DEFAULT_DB_NAME || 'default' }, `${window.GLOBAL_CONFIG.DEFAULT_DB_NAME || 'default'} (Default)`),
            // Map available databases
            availableDbs.map((db) => e(MenuItem, { key: db, value: db }, db))
          )
        ),
      // Existing controls follow...
      e(FormControlLabel, {
        control: e(Switch, {
          id: "optimizeToggle",
          checked: optimize, // Use prop
          onChange: handleOptimizeChange, // Calls prop function
          size: "small",
        }),
        label: e(Typography, { variant: "body2" }, "Optimize Response"),
        sx: { mb: 1, display: "flex", justifyContent: "space-between", ml: 0 },
      }),
      e(FormControlLabel, {
        control: e(Switch, {
          id: "hybridToggle",
          checked: hybrid, // Use prop
          onChange: handleHybridChange, // Calls prop function
          size: "small",
        }),
        label: e(Typography, { variant: "body2" }, "Hybrid Mode"),
        sx: { mb: 2, display: "flex", justifyContent: "space-between", ml: 0 },
      }),
      e(
        Button,
        {
          id: "openLlmSettingsBtn",
          variant: "outlined",
          fullWidth: true,
          onClick: handleOpenLlmSettings,
          startIcon: e(Icon, null, "settings"),
        },
        "Configure LLM"
      )
    ),
    // Chat History
    e(Divider),
    e(
      Box,
      { sx: { p: 2, flexGrow: 0 } },
      e(Typography, { variant: "subtitle1", component: "h2" }, "Chat History")
    ),
    e(
      Box,
      { sx: { flexGrow: 1, overflowY: "auto", minHeight: 100 } },
      e(
        List,
        { dense: true },
        chatHistory.length === 0 &&
          e(ListItem, null, e(ListItemText, { primary: "No chats yet..." })),
        chatHistory.map((chat) =>
          e(
            ListItem,
            {
              key: chat.id,
              button: true,
              selected: chat.id === activeChatId,
              onClick: () => handleSelectChat(chat.id),
              secondaryAction: e(
                Tooltip,
                { title: "Delete Chat" },
                e(
                  IconButton,
                  {
                    edge: "end",
                    "aria-label": "delete",
                    size: "small",
                    onClick: (event) => handleDeleteChat(event, chat.id),
                  },
                  e(Icon, { sx: { fontSize: "1rem" } }, "delete_outline")
                )
              ),
            },
            e(ListItemText, { primary: chat.title, sx: { pr: 4 } }) // Padding for delete icon
          )
        )
      )
    ),
    // Footer
    e(Divider),
    e(
      Box,
      { sx: { p: 2 } },
      e(
        Button,
        {
          variant: "contained",
          fullWidth: true,
          onClick: handleNewChat,
          startIcon: e(Icon, null, "add_comment"),
        },
        "New Chat"
      )
    )
  );

  return e(
    Drawer,
    {
      variant: "permanent",
      anchor: "left",
      sx: {
        width: SIDEBAR_WIDTH,
        flexShrink: 0,
        "& .MuiDrawer-paper": { width: SIDEBAR_WIDTH, boxSizing: "border-box" },
      },
    },
    drawerContent
  );
}

// Expose modal control globally (if needed)
// Note: This assumes the button exists in the DOM when Sidebar is defined.
// Consider alternative approaches if timing is an issue.
window.llmSettingsModalControl = {
  open: () => {
    const btn = document.getElementById("openLlmSettingsBtn");
    if (btn) {
      btn.click();
    } else {
      console.warn("Sidebar could not find openLlmSettingsBtn to click.");
    }
  },
};

console.log("Sidebar.js component defined.");
