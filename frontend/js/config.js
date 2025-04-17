// frontend/js/config.js

// Initialize markdown-it
const md = window.markdownit({
  html: true,
  linkify: true,
  typographer: true,
  highlight: function (str, lang) {
    if (lang && Prism.languages[lang]) {
      try {
        return `<pre class="line-numbers" data-language="${lang}"><code class="language-${lang}">${Prism.highlight(
          str,
          Prism.languages[lang],
          lang
        )}</code></pre>`;
      } catch (__) {}
    }
    // Use escapeHtml for plaintext or if language is not found/supported
    return `<pre class="line-numbers" data-language="plaintext"><code class="language-plaintext">${md.utils.escapeHtml(
      str
    )}</code></pre>`;
  },
});

// Add Prism line numbers plugin initialization if not already handled by auto-loader
// Prism.plugins.lineNumbers.initialize(); // Might be needed depending on Prism setup
