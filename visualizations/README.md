# ISE 5406 — Interactive Visualizations

Interactive demos for ISE 5405/5406:

- **Neural Network Visualization** — 8x8 digit classifier with live forward/backprop
- **First-Order Optimization Methods** — gradient descent, momentum, Adam on 2D test functions
- **Anatomy of a Step** — step-by-step walkthrough of one optimizer iteration
- **NN & Backpropagation Treatise** — standalone interactive document

## For students

Just open the live site (link in the repo description on GitHub).

## Running locally

```bash
npm install
npm run dev
```

Then open http://localhost:5173.

## Deploying

Pushes to `main` auto-deploy to GitHub Pages via `.github/workflows/deploy.yml`.

To enable Pages on a fresh repo:

1. Push the code to GitHub.
2. Go to **Settings → Pages**.
3. Under **Source**, select **GitHub Actions**.
4. Push to `main` (or run the workflow manually) to trigger a deploy.
