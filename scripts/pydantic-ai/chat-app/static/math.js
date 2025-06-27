export function renderMathWithRetry(element, retries = 10) {
  if (window.renderMathInElement) {
    window.renderMathInElement(element, {
      delimiters: [
        { left: "$$", right: "$$", display: true },
        { left: "$", right: "$", display: false }
      ],
      ignoredTags: []
    });
  } else if (retries > 0) {
    setTimeout(() => renderMathWithRetry(element, retries - 1), 200);
  }
} 