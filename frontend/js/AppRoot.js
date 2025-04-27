// AppRoot.js - Updated

// Removed MaterialUI definitions - rely on globals from app.js
// darkTheme definition removed - moved to app.js to be globally available
// AppRoot relies on darkTheme being defined globally before this script runs

function AppRoot() {
  const { useState, useEffect, useCallback, createElement: e } = React;

  // --- State for Query Settings ---
  const [queryMode, setQueryMode] = useState("rag");
  const [optimize, setOptimize] = useState(false);
  const [hybrid, setHybrid] = useState(false);
  const [selectedDbName, setSelectedDbName] = useState(
    window.GLOBAL_CONFIG?.DEFAULT_DB_NAME || "default"
  ); // Add state for DB name

  // --- Load settings from localStorage on mount ---
  useEffect(() => {
    const savedQueryMode =
      localStorage.getItem("uiSettings_queryMode") || "rag";
    const savedOptimize =
      localStorage.getItem("uiSettings_optimize") === "true";
    const savedHybrid = localStorage.getItem("uiSettings_hybrid") === "true";
    const savedDbName =
      localStorage.getItem("uiSettings_selectedDbName") ||
      window.GLOBAL_CONFIG?.DEFAULT_DB_NAME ||
      "default";

    setQueryMode(savedQueryMode);
    setOptimize(savedOptimize);
    setHybrid(savedHybrid);
    setSelectedDbName(savedDbName); // Load saved DB name
  }, []); // Empty dependency array ensures this runs only once on mount

  // --- Save settings to localStorage on change ---
  useEffect(() => {
    localStorage.setItem("uiSettings_queryMode", queryMode);
  }, [queryMode]);
  useEffect(() => {
    localStorage.setItem("uiSettings_optimize", optimize);
  }, [optimize]);
  useEffect(() => {
    localStorage.setItem("uiSettings_hybrid", hybrid);
  }, [hybrid]);
  useEffect(() => {
    localStorage.setItem("uiSettings_selectedDbName", selectedDbName);
  }, [selectedDbName]); // Save DB name on change

  // --- Handlers to update state ---
  const handleQueryModeChange = useCallback((newMode) => {
    setQueryMode(newMode);
  }, []);
  const handleOptimizeChange = useCallback((newOptimize) => {
    setOptimize(newOptimize);
  }, []);
  const handleHybridChange = useCallback((newHybrid) => {
    setHybrid(newHybrid);
  }, []);
  const handleDbNameChange = useCallback((newDbName) => {
    setSelectedDbName(newDbName);
  }, []); // Handler for DB name change

  return e(
    ThemeProvider,
    { theme: darkTheme },
    e(CssBaseline),
    // Main application container using Box
    e(
      Box,
      { className: "app-container" },
      // Render Sidebar and MainContent components directly
      // Pass state and handlers to Sidebar
      e(Sidebar, {
        queryMode,
        optimize,
        hybrid,
        selectedDbName, // Pass state down
        onQueryModeChange: handleQueryModeChange,
        onOptimizeChange: handleOptimizeChange,
        onHybridChange: handleHybridChange,
        onDbNameChange: handleDbNameChange, // Pass handler down
      }),
      // Pass state to MainContent
      e(MainContent, {
        queryMode,
        optimize,
        hybrid,
        selectedDbName, // Pass state down
      })
    )
  );
}

// Mount the AppRoot component after the DOM is loaded
document.addEventListener("DOMContentLoaded", () => {
  const rootContainer = document.getElementById("root");
  if (rootContainer) {
    // Ensure React, ReactDOM, MaterialUI, and component functions (AppRoot, Sidebar, MainContent, etc.)
    // are available globally before this line executes.
    // Also ensure React and ReactDOM are loaded before MaterialUI.
    const { createElement: e } = React; // Define 'e' locally just before use
    // Use React 18 createRoot API
    const root = ReactDOM.createRoot(rootContainer);
    root.render(e(AppRoot));
    console.log("AppRoot mounted using createRoot.");
  } else {
    console.error(
      "Fatal Error: Root container (#root) not found in index.html!"
    );
  }
});
