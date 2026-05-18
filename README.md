# MLP Prediction Model

A Vercel-ready customer frequency prediction app.

The original Streamlit interface has been converted into a static web app that runs the trained Keras MLP directly in the browser. The model weights are extracted from `model.h5` into `model-weights.js`, and the JavaScript forward pass mirrors the saved network:

- Input: `Recency`, `Monetary`, `AvgQuantity`
- Hidden layer: Dense 64 with ReLU
- Hidden layer: Dense 32 with ReLU
- Output layer: Dense 1 with sigmoid
- Threshold: `0.5`

## Project Files

- `index.html` - App shell and prediction UI
- `styles.css` - Responsive Vercel app styling
- `app.js` - Lightweight neural-network inference
- `model-weights.js` - Extracted trained weights from the Keras `.h5` model
- `vercel.json` - Static hosting configuration

## Run Locally

```bash
python -m http.server 4173
```

Then open:

```text
http://localhost:4173
```

## Deploy

This project can be deployed as a static Vercel site from the repository root.
