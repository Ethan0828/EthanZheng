# ResearchLog — Personal Technical Blog

A lightweight static blog hosted on GitHub Pages. Supports Markdown and PDF articles, custom tags, and full-text search.

## 🚀 Quick Setup

### 1. Fork / Clone

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
```

### 2. Enable GitHub Pages

1. Go to your repo → **Settings** → **Pages**
2. Under **Source**, select **GitHub Actions**
3. Push any commit to `main` — the site will be live at:
   `https://YOUR_USERNAME.github.io/YOUR_REPO_NAME/`

### 3. Customize your blog title

Edit `index.html`, find this line and change it:

```html
<div class="site-title">Research<span>.</span>Log</div>
<div class="site-subtitle">// Personal Technical Articles</div>
```

---

## 📝 Adding Articles

### Method A: Manual (recommended)

1. **Add your file** to the `articles/` folder:
   - Markdown: `articles/my-article.md`
   - PDF: `articles/my-paper.pdf`

2. **Edit `data/articles.json`** and append an entry:

```json
{
  "id": "my-article",
  "title": "My Article Title",
  "file": "articles/my-article.md",
  "type": "md",
  "date": "2026-04-10",
  "tags": ["tag1", "tag2"],
  "keywords": ["keyword1", "keyword2", "keyword3"],
  "excerpt": "A short description shown on the card."
}
```

3. **Push to GitHub** — deploy is automatic.

### Method B: Using the UI

1. Open your blog site
2. Click **New Article** button (top right)
3. Fill in the form — it will generate the JSON entry for you
4. Copy the JSON and paste it into `data/articles.json`
5. Upload your file to `articles/` and push

---

## 📁 File Structure

```
your-blog/
├── index.html              # Main listing page (search + filters)
├── viewer.html             # Article reader (MD + PDF)
├── data/
│   └── articles.json       # ← Edit this to add/remove articles
├── articles/               # ← Put your .md and .pdf files here
│   └── spectral-gnn-intro.md
├── .github/
│   └── workflows/
│       └── deploy.yml      # Auto-deploy on push to main
└── README.md
```

---

## 🔍 Search & Tags

- **Search**: fuzzy-matches across title, keywords, excerpt, and tags using [Fuse.js](https://fusejs.io/)
- **Tag filter**: click any tag pill to filter articles; hold Ctrl/Cmd to multi-select
- **Type filter**: toggle between All / Markdown / PDF

---

## 📄 articles.json Schema

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | string | ✅ | Unique slug (used in URL: `viewer.html?id=...`) |
| `title` | string | ✅ | Article title |
| `file` | string | ✅ | Path to file, e.g. `articles/foo.md` |
| `type` | `"md"` or `"pdf"` | ✅ | File type |
| `date` | string | ✅ | ISO date, e.g. `2026-04-10` |
| `tags` | string[] | ✅ | Category tags (shown as filter pills) |
| `keywords` | string[] | ✅ | Search keywords (shown on article page) |
| `excerpt` | string | ✅ | Short description for the card |

---

## 🎨 Customization

### Change site name
Edit the `<div class="site-title">` in `index.html`.

### Change accent color
In both HTML files, find `:root` and change `--accent`:
```css
--accent: #cba6f7;   /* purple (default) */
--accent: #89b4fa;   /* blue */
--accent: #a6e3a1;   /* green */
```

### Add math rendering (KaTeX)
Add to `<head>` in `viewer.html`:
```html
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css" />
<script src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js"></script>
```
Then after `hljs.highlightAll()` in viewer.html:
```javascript
renderMathInElement(document.getElementById('mdContent'), {
  delimiters: [
    { left: '$$', right: '$$', display: true },
    { left: '$', right: '$', display: false }
  ]
});
```
