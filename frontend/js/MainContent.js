function MainContent({ queryMode, optimize, hybrid }) {
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
    e(InputArea, { queryMode, optimize, hybrid }) // Pass props down
  );
}

console.log("MainContent.js component defined.");
