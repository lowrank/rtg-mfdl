window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"], ["$", "$"]],
    displayMath: [["\\[", "\\]"], ["$$", "$$"]],
    processEscapes: true,
    processEnvironments: true,
    packages: {'[+]': ['boldsymbol']}
  }
};

document.addEventListener("DOMContentLoaded", function() {
  if (typeof MathJax !== 'undefined' && MathJax.typesetPromise) {
    MathJax.typesetPromise();
  }
});

// Support for MkDocs Material instant loading (SPA-like navigation)
if (typeof document$ !== "undefined") {
  document$.subscribe(function() {
    if (typeof MathJax !== 'undefined' && MathJax.typesetPromise) {
      MathJax.typesetPromise();
    }
  });
}
