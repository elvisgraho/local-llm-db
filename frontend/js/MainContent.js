function MainContent({ queryMode, optimize, hybrid, selectedDbName }) {
  // Add selectedDbName prop
  // Accept props
  return e(
    Box,
    {
      id: "mainContentMuiWrapper",
      sx: {
        display: "flex",
        flexDirection: "column",
        flexGrow: 1,
        height: "100vh",
        overflow: "hidden",
        bgcolor: "background.default",
      },
    },
    e(ChatArea),
    e(InputArea, { queryMode, optimize, hybrid, selectedDbName }) // Pass selectedDbName down
  );
}

console.log("MainContent.js component defined.");
