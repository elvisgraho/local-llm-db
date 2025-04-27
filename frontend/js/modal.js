// Relies on global React hooks (useState, useEffect, useCallback, createElement as e)
// Relies on global MaterialUI components (Dialog, DialogTitle, DialogContent, DialogActions, Button, TextField, FormControl, FormControlLabel, RadioGroup, Radio, Select, MenuItem, InputLabel, Box, CircularProgress, Alert, Typography, IconButton, Tooltip)
// Relies on global Icon helper
// All defined in app.js

function LlmSettingsModal() {
  // Define Icon helper locally
  const Icon = ({ children }) =>
    e("span", { className: "material-icons" }, children);

  const [open, setOpen] = useState(false);
  const [preferredProvider, setPreferredProvider] = useState("local"); // User's saved preference
  const [currentProvider, setCurrentProvider] = useState("local"); // Provider currently being viewed/configured in modal

  // Provider-specific states
  const [apiKeys, setApiKeys] = useState({ local: null, gemini: "" });
  const [modelsByProvider, setModelsByProvider] = useState({
    local: [],
    gemini: [],
  });
  const [selectedModels, setSelectedModels] = useState({
    local: "",
    gemini: "",
  });
  const [isLoading, setIsLoading] = useState({ local: false, gemini: false });
  const [errors, setErrors] = useState({ local: null, gemini: null });
  const [geminiKeyNeeded, setGeminiKeyNeeded] = useState(false);

  const [isSaving, setIsSaving] = useState(false);
  const [saveSuccess, setSaveSuccess] = useState(false);

  // --- Event Handlers ---

  const handleClose = useCallback(() => {
    setOpen(false);
    setIsSaving(false);
  }, []);

  const handleApiKeyChange = (event) => {
    const newApiKey = event.target.value;
    setApiKeys((prev) => ({ ...prev, [currentProvider]: newApiKey }));
    if (currentProvider === "gemini") {
      setGeminiKeyNeeded(false);
      setErrors((prev) => ({ ...prev, gemini: null }));
    }
  };

  const handleModelChange = (event) => {
    const newModel = event.target.value;
    setSelectedModels((prev) => ({ ...prev, [currentProvider]: newModel }));
  };

  // Removed handleContextLengthChange

  // --- Data Fetching & Loading ---

  const fetchModels = useCallback(async (providerToFetch, apiKeyToUse) => {
    setIsLoading((prev) => ({ ...prev, [providerToFetch]: true }));
    setErrors((prev) => ({ ...prev, [providerToFetch]: null }));
    setGeminiKeyNeeded(false);

    let effectiveApiKey = null;
    if (providerToFetch === "gemini") {
      effectiveApiKey = apiKeyToUse?.trim();
      if (!effectiveApiKey) {
        console.warn("Gemini API Key missing, skipping fetch.");
        setModelsByProvider((prev) => ({ ...prev, [providerToFetch]: [] }));
        setSelectedModels((prev) => ({ ...prev, [providerToFetch]: "" }));
        setIsLoading((prev) => ({ ...prev, [providerToFetch]: false }));
        setGeminiKeyNeeded(true);
        return;
      }
    }

    try {
      console.log(`Fetching models for provider: ${providerToFetch}`);
      const response = await fetch(`/api/models/${providerToFetch}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(
          providerToFetch === "gemini" ? { apiKey: effectiveApiKey } : {}
        ),
      });

      const data = await response.json();

      if (!response.ok || data.status === "error") {
        throw new Error(data.error || `HTTP error ${response.status}`);
      }

      if (data.models && data.models.length > 0) {
        setModelsByProvider((prev) => ({
          ...prev,
          [providerToFetch]: data.models,
        }));
        // Attempt to re-select saved model for *this* provider
        const configKey = `llmConfig_${providerToFetch}`;
        const savedConfig = localStorage.getItem(configKey);
        let savedModelName = "";
        if (savedConfig) {
          try {
            savedModelName = JSON.parse(savedConfig).modelName;
          } catch (e) {
            console.error("Failed to parse saved config for re-selection:", e);
          }
        }
        if (savedModelName && data.models.includes(savedModelName)) {
          setSelectedModels((prev) => ({
            ...prev,
            [providerToFetch]: savedModelName,
          }));
          console.log(
            `Re-selected saved model for ${providerToFetch}: ${savedModelName}`
          );
        } else {
          setSelectedModels((prev) => ({ ...prev, [providerToFetch]: "" }));
          if (savedModelName)
            console.warn(
              `Saved model "${savedModelName}" not found for ${providerToFetch}.`
            );
        }
      } else {
        setErrors((prev) => ({
          ...prev,
          [providerToFetch]: `No models found for ${providerToFetch}. Check server/API key.`,
        }));
        setModelsByProvider((prev) => ({ ...prev, [providerToFetch]: [] }));
        setSelectedModels((prev) => ({ ...prev, [providerToFetch]: "" }));
      }
    } catch (error) {
      console.error(`Error fetching models for ${providerToFetch}:`, error);
      setErrors((prev) => ({
        ...prev,
        [providerToFetch]: `Error fetching models: ${error.message}`,
      }));
      setModelsByProvider((prev) => ({ ...prev, [providerToFetch]: [] }));
      setSelectedModels((prev) => ({ ...prev, [providerToFetch]: "" }));
    } finally {
      setIsLoading((prev) => ({ ...prev, [providerToFetch]: false }));
    }
  }, []);

  const loadSettingsForProvider = useCallback(
    (providerToLoad) => {
      const configKey = `llmConfig_${providerToLoad}`;
      const savedConfig = localStorage.getItem(configKey);
      let apiKeyFromStorage = providerToLoad === "gemini" ? "" : null;
      let modelNameFromStorage = "";
      // Removed contextLengthFromStorage

      if (savedConfig) {
        try {
          const llmConfig = JSON.parse(savedConfig);
          console.log(`Loading settings for ${providerToLoad}:`, llmConfig);
          if (providerToLoad === "gemini" && llmConfig.apiKey) {
            apiKeyFromStorage = llmConfig.apiKey;
          }
          modelNameFromStorage = llmConfig.modelName || "";
          // Removed context length loading
        } catch (e) {
          console.error(
            `Failed to parse saved LLM config for ${providerToLoad}:`,
            e
          );
        }
      } else {
        console.log(`No saved settings for ${providerToLoad}.`);
      }

      setApiKeys((prev) => ({ ...prev, [providerToLoad]: apiKeyFromStorage }));
      setSelectedModels((prev) => ({
        ...prev,
        [providerToLoad]: modelNameFromStorage,
      }));
      // Removed context length state setting

      fetchModels(providerToLoad, apiKeyFromStorage);
    },
    [fetchModels]
  );

  useEffect(() => {
    const savedPreference =
      localStorage.getItem("preferredLlmProvider") || "local";
    setPreferredProvider(savedPreference);
    console.log("Initial preferred provider loaded:", savedPreference);
  }, []);

  const handleOpen = useCallback(() => {
    const providerToOpen = preferredProvider;
    setCurrentProvider(providerToOpen);
    loadSettingsForProvider(providerToOpen);
    setOpen(true);
    setSaveSuccess(false);
    setErrors({ local: null, gemini: null });
  }, [loadSettingsForProvider, preferredProvider]);

  const handlePreferredProviderChange = useCallback(
    (event) => {
      const newPreference = event.target.value;
      console.log("Preferred provider changed to:", newPreference);
      setPreferredProvider(newPreference);
      localStorage.setItem("preferredLlmProvider", newPreference);

      setCurrentProvider(newPreference);
      if (
        !modelsByProvider[newPreference] ||
        modelsByProvider[newPreference].length === 0
      ) {
        loadSettingsForProvider(newPreference);
      }
    },
    [modelsByProvider, loadSettingsForProvider]
  );

  const handleRefreshModels = useCallback(() => {
    fetchModels(currentProvider, apiKeys[currentProvider]);
  }, [currentProvider, apiKeys, fetchModels]);

  const handleSave = useCallback(() => {
    const currentApiKey = apiKeys[currentProvider];
    const currentSelectedModel = selectedModels[currentProvider];
    const currentModels = modelsByProvider[currentProvider];
    // Removed currentContextLength

    if (currentProvider === "gemini" && !currentApiKey?.trim()) {
      alert("Please enter your Gemini API Key before saving.");
      return;
    }

    if (!currentSelectedModel && currentModels && currentModels.length > 0) {
      alert("Please select a model from the list.");
      return;
    }

    // Removed context length validation

    setIsSaving(true);
    setSaveSuccess(false);

    const llmConfig = {
      provider: currentProvider,
      modelName: currentSelectedModel,
      apiKey: currentProvider === "gemini" ? currentApiKey?.trim() : null,
      // Removed contextLength from saved config
    };

    const configKey = `llmConfig_${currentProvider}`;
    console.log(`Saving LLM config to ${configKey}:`, llmConfig);
    localStorage.setItem(configKey, JSON.stringify(llmConfig));

    // Preserve other provider's config
    const otherProvider = currentProvider === "local" ? "gemini" : "local";
    const otherConfigKey = `llmConfig_${otherProvider}`;
    const otherSavedConfig = localStorage.getItem(otherConfigKey);
    if (otherSavedConfig) {
      try {
        const otherLlmConfig = JSON.parse(otherSavedConfig);
        otherLlmConfig.provider = otherProvider;
        // Removed contextLength preservation logic
        localStorage.setItem(otherConfigKey, JSON.stringify(otherLlmConfig));
        console.log(`Preserved existing config for ${otherProvider}`);
      } catch (e) {
        console.error(
          `Failed to parse/preserve config for ${otherProvider}:`,
          e
        );
      }
    }

    setTimeout(() => {
      setIsSaving(false);
      setSaveSuccess(true);
      console.log(`LLM Settings Saved to ${configKey}`);
      setTimeout(handleClose, 1000);
    }, 500);
  }, [currentProvider, apiKeys, selectedModels, modelsByProvider, handleClose]); // Removed contextLengths dependency

  useEffect(() => {
    const openBtn = document.getElementById("openLlmSettingsBtn");
    if (openBtn) {
      const openHandler = handleOpen;
      openBtn.addEventListener("click", openHandler);
      return () => openBtn.removeEventListener("click", openHandler);
    }
  }, [handleOpen]);

  // --- Render Logic ---
  const currentApiKey = apiKeys[currentProvider] ?? "";
  const currentModels = modelsByProvider[currentProvider] || [];
  const currentSelectedModel = selectedModels[currentProvider] || "";
  // Removed currentContextLengthValue
  const currentIsLoading = isLoading[currentProvider] || false;
  const currentError = errors[currentProvider] || null;
  const isRefreshDisabled =
    currentIsLoading || (currentProvider === "gemini" && !currentApiKey.trim());

  return e(
    Dialog,
    { open: open, onClose: handleClose, maxWidth: "sm", fullWidth: true },
    e(DialogTitle, null, "LLM Configuration"),
    e(
      DialogContent,
      { dividers: true },
      // Preferred Provider Selection (Dropdown)
      e(
        FormControl,
        { fullWidth: true, margin: "normal", variant: "outlined" },
        e(
          InputLabel,
          { id: "preferred-provider-label" },
          "Preferred LLM Provider"
        ),
        e(
          Select,
          {
            labelId: "preferred-provider-label",
            id: "preferredLlmProviderSelect",
            value: preferredProvider,
            onChange: handlePreferredProviderChange,
            label: "Preferred LLM Provider",
          },
          e(MenuItem, { value: "local" }, "Local LLM"),
          e(MenuItem, { value: "gemini" }, "Gemini API")
        ),
        e(
          Typography,
          { variant: "caption", display: "block", sx: { mt: 0.5, ml: 1.5 } },
          "Select your default provider. Settings below apply to the selected provider."
        )
      ),

      // --- Configuration Section for the 'currentProvider' ---
      e(
        Typography,
        {
          variant: "subtitle1",
          gutterBottom: true,
          sx: {
            mt: 2,
            borderTop: "1px solid rgba(255, 255, 255, 0.12)",
            pt: 2,
          },
        },
        `Configure: ${currentProvider === "local" ? "Local LLM" : "Gemini API"}`
      ),

      // Gemini API Key Input (Only shown when configuring Gemini)
      currentProvider === "gemini" &&
        e(TextField, {
          label: "Gemini API Key",
          type: "password",
          value: currentApiKey,
          onChange: handleApiKeyChange,
          fullWidth: true,
          margin: "normal",
          variant: "outlined",
          required: true,
        }),
      // Model Selection Dropdown + Refresh Button
      e(
        Box,
        { sx: { display: "flex", alignItems: "center", gap: 1, mt: 2, mb: 1 } },
        e(
          FormControl,
          {
            fullWidth: true,
            variant: "outlined",
            margin: "normal",
            required: currentModels.length > 0,
          },
          e(InputLabel, { id: "model-select-label" }, "Model Name"),
          e(
            Select,
            {
              labelId: "model-select-label",
              id: "llmModelNameModal",
              value: currentSelectedModel,
              onChange: handleModelChange,
              label: "Model Name",
              disabled: currentIsLoading || currentModels.length === 0,
            },
            currentIsLoading &&
              e(
                MenuItem,
                { value: "", disabled: true },
                e(
                  Box,
                  { sx: { display: "flex", alignItems: "center", gap: 1 } },
                  e(CircularProgress, { size: 20 }),
                  "Loading models..."
                )
              ),
            !currentIsLoading &&
              currentProvider === "gemini" &&
              geminiKeyNeeded &&
              e(
                MenuItem,
                { value: "", disabled: true },
                "Enter API Key and Refresh"
              ),
            !currentIsLoading &&
              !geminiKeyNeeded &&
              currentModels.length === 0 &&
              !currentError &&
              e(MenuItem, { value: "", disabled: true }, "No models available"),
            !currentIsLoading &&
              !geminiKeyNeeded &&
              currentError &&
              e(
                MenuItem,
                { value: "", disabled: true },
                "Error loading models"
              ),
            !currentIsLoading &&
              currentModels.map((modelId) =>
                e(MenuItem, { key: modelId, value: modelId }, modelId)
              )
          )
        ),
        e(
          Tooltip,
          { title: "Refresh model list" },
          e(
            "span",
            null,
            e(
              IconButton,
              {
                onClick: handleRefreshModels,
                disabled: isRefreshDisabled,
              },
              e(Icon, null, "refresh")
            )
          )
        )
      ),
      // Removed Context Length Input

      // Error Alert for the current provider
      currentError &&
        e(Alert, { severity: "error", sx: { mt: 1 } }, currentError)
    ),
    // Dialog Actions (Buttons)
    e(
      DialogActions,
      null,
      e(Button, { onClick: handleClose, color: "secondary" }, "Cancel"),
      e(
        Button,
        {
          onClick: handleSave,
          color: "primary",
          variant: "contained",
          disabled: isSaving || currentIsLoading,
          startIcon: isSaving
            ? e(CircularProgress, { size: 20, color: "inherit" })
            : null,
        },
        isSaving ? "Saving..." : saveSuccess ? "Saved!" : "Save Settings"
      )
    )
  );
}

// Mount the React component after the DOM is fully loaded
document.addEventListener("DOMContentLoaded", () => {
  const domContainer = document.querySelector("#llmSettingsModalContainer");
  if (domContainer) {
    // Define 'e' locally for this mounting block and use React 18 createRoot API
    const { createElement: e } = React;
    const root = ReactDOM.createRoot(domContainer);
    // Define darkTheme here or ensure it's globally available before this point
    const darkTheme = createTheme({
      /* ... theme definition ... */
    });
    root.render(
      e(MaterialUI.ThemeProvider, { theme: darkTheme }, e(LlmSettingsModal))
    );
  } else {
    console.error("LLM Settings Modal container not found.");
  }
});

console.log("modal.js loaded and LlmSettingsModal defined.");
