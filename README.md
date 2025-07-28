# 📚 FungiTastic Documentation

This directory contains the **MkDocs**-powered documentation for the FungiTastic dataset and benchmark.  
It uses the [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) theme for a modern, searchable, and easy-to-navigate documentation site.

---

## 🚀 Quickstart

### 1. Install MkDocs and Dependencies

First, make sure you have **Python 3.7+**.

We recommend creating a virtual environment (optional, but best practice):

```bash
python -m venv venv
source venv/bin/activate   # On Linux/macOS
venv\Scriptsctivate      # On Windows
```

Now install MkDocs and the Material theme:

```bash
pip install mkdocs-material pymdown-extensions
```

---

### 2. Preview the Documentation Locally

From the root directory of the repository (where `mkdocs.yml` is located), run:

```bash
mkdocs serve
```

Then open your browser and go to [http://localhost:8000](http://localhost:8000).  
The site will automatically reload as you edit Markdown files in the `docs/` folder.

---

### 3. Build the Documentation

To generate a static site in the `site/` directory:

```bash
mkdocs build
```

---

### 4. Publish to GitHub Pages

You can deploy your docs to GitHub Pages in a single command:

```bash
mkdocs gh-deploy
```

This will:

- Build the documentation.
- Push the result to a `gh-pages` branch.
- Serve your site at:  
  `https://<your-username>.github.io/<repo-name>/`

You may need to enable GitHub Pages in your repo settings (set the source to `gh-pages`).

---

## 🛠️ Editing the Docs

- All documentation pages are Markdown files in the `docs/` folder.
- The structure and navigation is defined in `mkdocs.yml`.
- Edit the `.md` files, save, and your site will update on the next build or serve.

---

## 🧰 Troubleshooting

- If `mkdocs` is not found, ensure your virtual environment is activated and installed in the right Python.
- For theme/extension problems, see the [Material for MkDocs documentation](https://squidfunk.github.io/mkdocs-material/).
- For more help, open a [GitHub Issue](https://github.com/bohemianvra/FungiTastic/issues).

---

## 🔗 Useful Links

- [MkDocs documentation](https://www.mkdocs.org/)
- [Material for MkDocs documentation](https://squidfunk.github.io/mkdocs-material/)
- [FungiTastic main repo](https://github.com/bohemianvra/FungiTastic/)

---

Happy documenting!
