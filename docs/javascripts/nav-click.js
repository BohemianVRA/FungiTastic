/**
 * Auto-open overview page when a top-level nav item is clicked
 * Works for "Datasets", "Benchmarks", etc.
 */
document.addEventListener("DOMContentLoaded", () => {
  const redirects = {
    "Dataset Overview": "/dataset/",           // link to docs/dataset.md
    "Benchmarks": "/benchmarks/",      // optional: docs/benchmarks.md
    "Baselines & Models": "/baselines_and_models/",
    "How to Use": "/usage/"
  };

  const navLinks = document.querySelectorAll(".md-nav__link");

  navLinks.forEach(link => {
    const label = link.textContent.trim();
    if (redirects[label]) {
      link.addEventListener("click", e => {
        // allow the sidebar to expand but also navigate
        setTimeout(() => { window.location.href = redirects[label]; }, 120);
      });
    }
  });
});