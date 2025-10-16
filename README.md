# CT vs MRI Analyzer (Demo)

This is a client-side demo that attempts to distinguish CT vs MRI images using simple feature extraction and a heuristic linear classifier. It is NOT medical advice and NOT for clinical use.

- Next.js App Router, TypeScript
- Runs fully in the browser, no server ML
- Heatmap is a variance-based visualization, not Grad-CAM

## Local dev

```bash
npm install
npm run dev
```

## Build

```bash
npm run build && npm start
```

## Deploy

The project is configured to deploy on Vercel. You can run:

```bash
vercel deploy --prod --yes --token $VERCEL_TOKEN --name agentic-530f45a0
```
