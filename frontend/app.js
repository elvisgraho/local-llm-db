// Global React and Material UI constants for components
const { useState, useEffect, useRef, useCallback, createElement: e } = React;
const {
  Box,
  Paper,
  Typography,
  Chip,
  CircularProgress,
  Alert,
  TextField,
  IconButton,
  FormControlLabel,
  Checkbox,
  Tooltip,
  Drawer,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Divider,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Switch,
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  RadioGroup,
  Radio,
  ThemeProvider,
  createTheme,
  CssBaseline,
} = MaterialUI;

// Define the dark theme globally *before* components that use it
const darkTheme = createTheme({
  palette: {
    mode: "dark",
    primary: { main: "#66bb6a" },
    secondary: { main: "#ba68c8" },
    background: { default: "#303030", paper: "#424242" },
    text: { primary: "#ffffff", secondary: "#c5c5d2" },
  },
  typography: { fontFamily: '"Roboto", "Helvetica", "Arial", sans-serif' },
  components: {},
});

// Helper for Material Icons (Globally available)
const Icon = ({ children, sx }) =>
  e("span", { className: "material-icons", style: sx }, children);

// --- Main Query Function ---
// Reverted to original signature, backend will handle context length and history truncation
window.sendQuery = async function (
  queryText,
  includeHistory,
  queryMode, // This is rag_type
  optimize,
  hybrid,
  selectedDbName // Added argument
) {
  // Start thinking indicator
  window.chatAreaControl?.startThinking();

  // Use queryText argument directly
  if (!queryText) {
    console.warn("Send Query: query_text is empty.");
    window.chatAreaControl?.stopThinking();
    return;
  }

  if (!window.chatManager?.getCurrentChatId()) {
    console.error("Send Query Error: Chat not initialized.");
    window.chatAreaControl?.stopThinking();
    alert("Error: Chat not ready. Please try again.");
    return;
  }

  // Add user message via chatManager (UI updates via ChatArea listener)
  const userMessage = {
    content: queryText,
    isUser: true,
    timestamp: new Date().toISOString(),
    messageId: `msg_${Date.now()}`, // Simple ID for UI purposes
  };
  window.chatManager.addMessageToHistory(userMessage);

  // Prepare request body
  let requestBody = {
    query_text: queryText,
    rag_type: queryMode, // Rename 'mode' to 'rag_type' for backend consistency
    db_name: selectedDbName,
    optimize: optimize,
    hybrid: hybrid,
    llm_config: {},
    conversation_history: [], // Send full history if included
  };

  // Load preferred LLM provider and its config from localStorage
  const preferredProvider =
    localStorage.getItem("preferredLlmProvider") || "local";
  const configKey = `llmConfig_${preferredProvider}`;
  const savedLlmConfig = localStorage.getItem(configKey);

  console.log(`Using preferred provider: ${preferredProvider}`);

  if (savedLlmConfig) {
    try {
      // Parse saved config. It might include contextLength for local provider.
      const parsedConfig = JSON.parse(savedLlmConfig);
      // Ensure provider matches preference, as saved config might be stale if preference changed
      parsedConfig.provider = preferredProvider;
      requestBody.llm_config = parsedConfig;
    } catch (e) {
      console.error(`Failed to parse saved LLM config from ${configKey}:`, e);
      requestBody.llm_config = { provider: preferredProvider };
    }
  } else {
    console.warn(
      `No specific config found for preferred provider ${preferredProvider}. Sending default.`
    );
    requestBody.llm_config = { provider: preferredProvider };
  }

  // Clean up potentially stale keys/models
  if (preferredProvider !== "gemini") {
    requestBody.llm_config.apiKey = null;
  }
  if (!requestBody.llm_config.modelName) {
    requestBody.llm_config.modelName = "";
  }

  // Include *full* conversation history if checked (backend will truncate)
  if (includeHistory) {
    const currentChat = window.chatManager.getCurrentChat();
    // Include history *before* the current user message
    if (currentChat?.messages.length > 1) {
      requestBody.conversation_history = currentChat.messages
        .slice(0, -1) // Exclude the user message just added
        .map((msg) => ({
          role: msg.isUser ? "user" : "assistant",
          content: msg.content || "", // Ensure content is string
        }));
    }
  }

  try {
    console.log("Sending query payload (full history):", requestBody);
    const response = await fetch("/api/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(requestBody),
    });

    window.chatAreaControl?.stopThinking();

    let responseData;
    try {
      responseData = await response.json();
    } catch (e) {
      throw new Error(
        `Server returned non-JSON response: ${response.status} ${response.statusText}`
      );
    }

    if (!response.ok || responseData.status === "error") {
      const errorMessage =
        responseData.error || `HTTP error ${response.status}`;
      console.error("Backend Error:", errorMessage, responseData);
      throw new Error(errorMessage);
    }

    const assistantResponse =
      responseData.data?.text || "No response text found.";
    const assistantMessage = {
      content: assistantResponse,
      isUser: false,
      timestamp: new Date().toISOString(),
      messageId: `msg_${Date.now()}_${Math.random()
        .toString(36)
        .substring(2, 7)}`, // Unique ID
      sources: responseData.data?.sources || [],
      // Display estimated context tokens if returned by backend
      estimatedContextTokens: responseData.data?.estimated_context_tokens,
    };
    window.chatManager.addMessageToHistory(assistantMessage);
    console.log("Assistant message added to history via chatManager.");

    // Handle special sections (remains the same)
    const specialSections = responseData.data?.special_sections;
    if (specialSections) {
      handleSpecialSections(specialSections);
    }
  } catch (error) {
    console.error("Error sending query:", error);
    window.chatAreaControl?.stopThinking();
    // Add error message to chat via chatManager
    const errorMessage = {
      content: `Error: ${error.message}`,
      isUser: false,
      timestamp: new Date().toISOString(),
      messageId: `msg_${Date.now()}`,
    };
    window.chatManager.addMessageToHistory(errorMessage);
  }
};

// --- Handle Special Sections ---
function handleSpecialSections(sections) {
  const chatContainer = document.getElementById("chatContainer");
  if (!chatContainer) {
    console.error("Special Sections Error: Chat container not found.");
    return;
  }

  const specialSectionsDiv = document.createElement("div");
  specialSectionsDiv.className =
    "special-section-container message assistant-message"; // Style like a message
  specialSectionsDiv.style.backgroundColor = "rgba(200, 200, 220, 0.1)"; // Slightly distinct background
  specialSectionsDiv.style.borderLeft = "3px solid #888";
  specialSectionsDiv.style.marginTop = "10px";

  const sectionHandlers = {
    sources: (content) => {
      if (!Array.isArray(content)) return "";
      const sourcesList = content
        .map(
          (src) =>
            `<li>${src.document || "Unknown"} (Score: ${(
              src.score || 0
            ).toFixed(2)})</li>`
        )
        .join("");
      return `<h6>Sources:</h6><ul>${sourcesList}</ul>`;
    },
    file_path: (content) => {
      if (!Array.isArray(content)) return "";
      const fileLinks = content
        .map((path) => {
          if (typeof path !== "string") return "";
          const escapedPath = path.replace(/\\/g, "\\\\").replace(/'/g, "\\'");
          const onclickHandler = window.uiUtils?.openLocalFile
            ? `window.uiUtils.openLocalFile('${escapedPath}')`
            : `console.error('openLocalFile function not found')`;
          return `<li><a href="#" onclick="${onclickHandler}; return false;">${path}</a></li>`;
        })
        .join("");
      return `<h6>Referenced Files:</h6><ul>${fileLinks}</ul>`;
    },
  };

  let htmlContent = "";
  for (const key in sections) {
    if (sectionHandlers[key]) {
      htmlContent += sectionHandlers[key](sections[key]);
    } else {
      console.warn(`Unknown special section type: ${key}`);
      try {
        const cleanKey = key.replace(/[^a-zA-Z0-9_]/g, "").replace(/_/g, " ");
        const safeJson = JSON.stringify(sections[key], null, 2)
          .replace(/</g, "<")
          .replace(/>/g, ">");
        htmlContent += `<h6>${cleanKey}:</h6><pre><code>${safeJson}</code></pre>`;
      } catch (e) {
        htmlContent += `<h6>${key
          .replace(/</g, "<")
          .replace(/>/g, ">")}:</h6><p>[Could not display content]</p>`;
      }
    }
  }

  if (htmlContent && window.DOMPurify) {
    specialSectionsDiv.innerHTML = DOMPurify.sanitize(htmlContent);
    chatContainer.appendChild(specialSectionsDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;

    window.Prism?.highlightAllUnder(specialSectionsDiv);
  } else if (htmlContent) {
    console.error("DOMPurify not available to sanitize special sections HTML.");
  }
}

// --- App Root Component ---
function AppRoot() {
  // darkTheme is now defined globally above

  // State for settings shared between Sidebar and InputArea
  const [queryMode, setQueryMode] = useState("rag");
  const [optimize, setOptimize] = useState(false);
  const [hybrid, setHybrid] = useState(false);
  const [selectedDbName, setSelectedDbName] = useState(
    window.GLOBAL_CONFIG?.DEFAULT_DB_NAME || "default"
  );

  // Load settings on initial mount
  useEffect(() => {
    const savedMode = localStorage.getItem("queryMode") || "rag";
    const savedOptimize = localStorage.getItem("optimize") === "true";
    const savedHybrid = localStorage.getItem("hybrid") === "true";
    const savedDbName =
      localStorage.getItem(`dbName_${savedMode}`) ||
      window.GLOBAL_CONFIG?.DEFAULT_DB_NAME ||
      "default";

    setQueryMode(savedMode);
    setOptimize(savedOptimize);
    setHybrid(savedHybrid);
    setSelectedDbName(savedDbName);

    console.log("AppRoot: Initial settings loaded", {
      mode: savedMode,
      optimize: savedOptimize,
      hybrid: savedHybrid,
      dbName: savedDbName,
    });
  }, []);

  // Save settings whenever they change
  useEffect(() => {
    localStorage.setItem("queryMode", queryMode);
    localStorage.setItem("optimize", optimize);
    localStorage.setItem("hybrid", hybrid);
    if (["rag", "kag", "lightrag"].includes(queryMode)) {
      localStorage.setItem(`dbName_${queryMode}`, selectedDbName);
    }
    console.log("AppRoot: Settings saved", {
      queryMode,
      optimize,
      hybrid,
      selectedDbName,
    });
  }, [queryMode, optimize, hybrid, selectedDbName]);

  // Handlers to update state
  const handleQueryModeChange = useCallback((newMode) => {
    setQueryMode(newMode);
    const savedDbForNewMode =
      localStorage.getItem(`dbName_${newMode}`) ||
      window.GLOBAL_CONFIG?.DEFAULT_DB_NAME ||
      "default";
    setSelectedDbName(savedDbForNewMode);
    console.log(`Mode changed to ${newMode}, loaded DB: ${savedDbForNewMode}`);
  }, []);

  const handleOptimizeChange = useCallback((newOptimize) => {
    setOptimize(newOptimize);
  }, []);

  const handleHybridChange = useCallback((newHybrid) => {
    setHybrid(newHybrid);
  }, []);

  const handleDbNameChange = useCallback(
    (newDbName) => {
      setSelectedDbName(newDbName);
      if (["rag", "kag", "lightrag"].includes(queryMode)) {
        localStorage.setItem(`dbName_${queryMode}`, newDbName);
      }
    },
    [queryMode]
  );

  return e(
    ThemeProvider,
    { theme: darkTheme }, // Apply the theme here
    e(CssBaseline),
    e(
      Box,
      { sx: { display: "flex", height: "100vh" } },
      e(Sidebar, {
        queryMode,
        optimize,
        hybrid,
        selectedDbName,
        onQueryModeChange: handleQueryModeChange,
        onOptimizeChange: handleOptimizeChange,
        onHybridChange: handleHybridChange,
        onDbNameChange: handleDbNameChange,
      }),
      e(
        Box,
        {
          component: "main",
          sx: {
            flexGrow: 1,
            display: "flex",
            flexDirection: "column",
            height: "100vh",
            overflow: "hidden",
          },
        },
        e(ChatArea),
        e(InputArea, { queryMode, optimize, hybrid, selectedDbName })
      )
    )
  );
}

// --- Mount App ---
document.addEventListener("DOMContentLoaded", () => {
  const domContainer = document.querySelector("#root");
  if (domContainer) {
    const root = ReactDOM.createRoot(domContainer);
    root.render(e(AppRoot));
    console.log("AppRoot mounted using createRoot."); // Moved log here
  } else {
    console.error("App Root container not found."); // Keep error log
  }
});

console.log("app.js initialized.");
