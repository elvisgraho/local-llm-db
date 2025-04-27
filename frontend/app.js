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

// Helper for Material Icons (Globally available)
const Icon = ({ children, sx }) =>
  e("span", { className: "material-icons", style: sx }, children);

// Define the dark theme globally (used by AppRoot and Modal)
const darkTheme = createTheme({
  palette: {
    mode: "dark",
    primary: {
      main: "#66bb6a", // MUI green
    },
    secondary: {
      main: "#ba68c8", // MUI purple
    },
    background: {
      default: "#303030",
      paper: "#424242",
    },
    text: {
      primary: "#ffffff",
      secondary: "#c5c5d2",
    },
  },
  typography: {
    fontFamily: '"Roboto", "Helvetica", "Arial", sans-serif',
  },
  components: {
    // Add global component overrides here if needed
  },
});

// --- Export Chat (Run after DOM is loaded) ---
document.addEventListener("DOMContentLoaded", () => {
  // Export button functionality removed as requested.
}); // End of document.addEventListener('DOMContentLoaded')

// --- Main Query Function ---
// --- Main Query Function ---
// Modified to accept settings and db_name as arguments
window.sendQuery = async function (
  queryText,
  includeHistory,
  queryMode, // This is now rag_type
  optimize,
  hybrid,
  selectedDbName // Added argument
) {
  // Start thinking indicator
  window.chatAreaControl?.startThinking();

  // --- Remove DOM reading for inputs and settings ---
  // const queryInput = document.getElementById("queryInput");
  // const includeHistoryCheckbox = document.getElementById("includeHistory");
  // const query = queryInput?.value.trim() || ""; // Use queryText argument
  // const includeHistory = includeHistoryCheckbox?.checked || false; // Use includeHistory argument

  // Use queryText argument directly
  if (!queryText) {
    window.chatAreaControl?.stopThinking();
    return;
  }

  if (!window.chatManager?.getCurrentChatId()) {
    console.error("Send Query Error: Chat not initialized.");
    window.chatAreaControl?.stopThinking();
    alert("Error: Chat not ready. Please try again.");
    return;
  }

  // --- Remove DOM reading for settings ---
  // const queryModeSelect = document.getElementById("queryMode");
  // const optimizeToggle = document.getElementById("optimizeToggle");
  // const hybridToggle = document.getElementById("hybridToggle");
  // const queryMode = queryModeSelect?.value || "rag"; // Use queryMode argument
  // const optimize = optimizeToggle?.checked || false; // Use optimize argument
  // const hybrid = hybridToggle?.checked || false; // Use hybrid argument

  // Add user message via chatManager (UI updates via ChatArea listener)
  const userMessage = {
    content: queryText, // Use queryText argument
    isUser: true,
    timestamp: new Date().toISOString(),
    messageId: `msg_${Date.now()}`,
  };
  window.chatManager.addMessageToHistory(userMessage);

  // Prepare request body
  let requestBody = {
    query_text: queryText, // Use queryText argument
    rag_type: queryMode, // Rename 'mode' to 'rag_type' for backend consistency
    db_name: selectedDbName, // Add db_name
    optimize: optimize, // Use optimize argument
    hybrid: hybrid,
    llm_config: {},
    conversation_history: [],
  };

  // Load preferred LLM provider and its config from localStorage
  const preferredProvider =
    localStorage.getItem("preferredLlmProvider") || "local";
  const configKey = `llmConfig_${preferredProvider}`;
  const savedLlmConfig = localStorage.getItem(configKey);

  console.log(`Using preferred provider: ${preferredProvider}`);

  if (savedLlmConfig) {
    try {
      requestBody.llm_config = JSON.parse(savedLlmConfig);
      // Ensure the provider field in the loaded config matches the preference
      // (This handles cases where config might be stale or manually edited)
      requestBody.llm_config.provider = preferredProvider;
    } catch (e) {
      console.error(`Failed to parse saved LLM config from ${configKey}:`, e);
      // Fallback to just the provider name if parsing fails
      requestBody.llm_config = { provider: preferredProvider };
    }
  } else {
    // If no specific config saved for the preferred provider, just send the provider name
    console.warn(
      `No specific config found for preferred provider ${preferredProvider}. Sending default.`
    );
    requestBody.llm_config = { provider: preferredProvider };
  }

  // Ensure API key is explicitly null if the preferred provider is not Gemini
  // (Cleans up potentially stale keys if user switches preference)
  if (preferredProvider !== "gemini") {
    requestBody.llm_config.apiKey = null;
  }
  // Ensure modelName is present, even if empty, for backend consistency
  if (!requestBody.llm_config.modelName) {
    requestBody.llm_config.modelName = "";
  }

  // Include conversation history if checked
  if (includeHistory) {
    const currentChat = window.chatManager.getCurrentChat();
    // Include history *before* the current user message
    if (currentChat?.messages.length > 1) {
      requestBody.conversation_history = currentChat.messages
        .slice(0, -1)
        .map((msg) => ({
          role: msg.isUser ? "user" : "assistant",
          content: msg.content,
        }));
    }
  }

  // Input clearing is handled by InputArea component

  try {
    console.log("Sending query:", requestBody);
    const response = await fetch("/api/query", {
      // Use relative path
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
      responseData.data?.text ||
      responseData.response ||
      "No response text found.";
    const assistantMessage = {
      content: assistantResponse,
      isUser: false,
      timestamp: new Date().toISOString(),
      // Ensure unique ID, Date.now() might collide on rapid responses
      messageId: `msg_${Date.now()}_${Math.random()
        .toString(36)
        .substring(2, 7)}`,
      sources: responseData.data?.sources || [], // Add sources here
    };
    window.chatManager.addMessageToHistory(assistantMessage);
    console.log("Assistant message should now be in history via chatManager."); // Log after adding

    // Handle special sections
    const specialSections =
      responseData.data?.special_sections || responseData.special_sections;
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
// Appends a separate div for special content like sources or file paths.
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
          .replace(/</g, "&lt;")
          .replace(/>/g, "&gt;");
        htmlContent += `<h6>${cleanKey}:</h6><pre><code>${safeJson}</code></pre>`;
      } catch (e) {
        htmlContent += `<h6>${key
          .replace(/</g, "&lt;")
          .replace(/>/g, "&gt;")}:</h6><p>[Could not display content]</p>`;
      }
    }
  }

  if (htmlContent && window.DOMPurify) {
    specialSectionsDiv.innerHTML = DOMPurify.sanitize(htmlContent);
    chatContainer.appendChild(specialSectionsDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;

    // Apply Prism highlighting and copy buttons
    window.Prism?.highlightAllUnder(specialSectionsDiv);
    // Rely on ChatArea/ChatMessage to re-run addCopyButtons after new message/section added
    // window.uiUtils?.addCopyButtons(); // Avoid potentially redundant global calls here
  } else if (htmlContent) {
    console.error("DOMPurify not available to sanitize special sections HTML.");
    // Avoid setting innerHTML without sanitization
  }
}

console.log("app.js initialized.");
