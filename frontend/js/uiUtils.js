// frontend/js/uiUtils.js - Refactored Utilities

// --- Utility Functions ---

// Rough estimation of tokens (4 characters per token on average)
function estimateTokens(text) {
  return Math.ceil((text || "").length / 4);
}

// --- Code Block Copy Button ---
// Note: This function relies on querying the DOM globally. It might need
// adaptation to work reliably within React components if timing issues arise.
// Consider passing the specific message element reference to this function in the future.
function addCopyButtons() {
  // Remove existing headers first to avoid duplication if called multiple times
  document
    .querySelectorAll(".code-block-header")
    .forEach((header) => header.remove());

  // Find code blocks (assuming they are rendered within elements having 'pre code')
  document.querySelectorAll("pre code").forEach((block) => {
    const pre = block.parentElement;
    // Check if it looks like a Prism-highlighted block
    // The 'line-numbers' class might not always be present depending on Prism setup
    if (pre && pre.tagName === "PRE") {
      // Try to get language from common attributes
      const language =
        block.className.match(/language-(\w+)/)?.[1] ||
        pre.getAttribute("data-language") ||
        "plaintext";

      const header = document.createElement("div");
      header.className = "code-block-header"; // Keep class for potential minimal styling

      const languageSpan = document.createElement("span");
      languageSpan.className = "code-block-language"; // Keep class
      languageSpan.textContent = language;

      const copyButton = document.createElement("button");
      copyButton.className = "copy-button"; // Keep class
      copyButton.textContent = "Copy";
      copyButton.title = "Copy code to clipboard";
      // Basic styling to make it look like a button without CSS
      copyButton.style.marginLeft = "10px";
      copyButton.style.padding = "2px 6px";
      copyButton.style.border = "1px solid #ccc";
      copyButton.style.borderRadius = "3px";
      copyButton.style.cursor = "pointer";
      copyButton.style.fontSize = "0.8em";

      copyButton.onclick = async () => {
        try {
          await navigator.clipboard.writeText(block.textContent);
          copyButton.textContent = "Copied!";
          copyButton.style.backgroundColor = "#90ee90"; // Light green feedback
          setTimeout(() => {
            copyButton.textContent = "Copy";
            copyButton.style.backgroundColor = "";
          }, 2000);
        } catch (err) {
          console.error("Failed to copy text:", err);
          copyButton.textContent = "Error";
          copyButton.style.backgroundColor = "#ffcccb"; // Light red feedback
          setTimeout(() => {
            copyButton.textContent = "Copy";
            copyButton.style.backgroundColor = "";
          }, 2000);
        }
      };

      header.appendChild(languageSpan);
      header.appendChild(copyButton);
      // Insert header before the code block within the pre tag
      if (!pre.querySelector(".code-block-header")) {
        // Avoid adding multiple headers
        pre.insertBefore(header, block);
      }
    }
  });
  // Prism initialization (highlighting and plugins like line numbers)
  // should ideally be handled within the React component's useEffect hook
  // after the code content is rendered to ensure correct timing.
  // Removing the global re-initialization attempt from here.
}

// --- Open Local File --- (Requires backend endpoint)
async function openLocalFile(filePath) {
  try {
    const response = await fetch("http://localhost:5000/open_file", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ file_path: filePath }),
    });

    if (!response.ok) {
      const errorData = await response
        .json()
        .catch(() => ({ error: "Unknown error occurred" }));
      throw new Error(
        errorData.error || `HTTP error! status: ${response.status}`
      );
    }

    const data = await response.json();
    if (data.status === "error") {
      throw new Error(data.error || "Backend reported an error opening file");
    }
    console.log("File open request sent successfully for:", filePath);
  } catch (error) {
    console.error("Error opening file:", error);
    alert(`Failed to open file: ${error.message}`);
  }
}

// --- Global Export ---
// Expose only the necessary functions globally
window.uiUtils = {
  estimateTokens,
  addCopyButtons,
  openLocalFile,
};

console.log("uiUtils.js refactored and loaded.");
