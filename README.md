# MarketMind

**AI-powered stock analysis: live market data, news sentiment, next-day price predictions and a conversational assistant in one dashboard.**

**Live demo:** https://marketmind-green.vercel.app/
*(The backend may take ~30 seconds to wake up on the first request.)*

Research model repo: [StockMarketModel](https://github.com/HimanshuJain-2004/StockMarketModel) (training code for the Refined REGCN)

---

## What it does

MarketMind gives retail investors a second, explainable opinion on the market:

- **Next-day predictions** for all 28 Dow Jones (DJIA) stocks, with a direction and confidence score
- **Company news with sentiment**, using a 3-provider fallback chain
- **AI assistant** that can look up prices, run the same prediction model as the dashboard, search the web, and explain results in plain English
- **Accounts** with email/password and Google sign-in, plus saved chat threads
- **Responsive UI** with dark and light themes

## How the prediction engine works

The predictor is a custom model, not a third-party API. It was designed, trained and benchmarked in a separate research repo.

1. **Variational Mode Decomposition (VMD)** splits each stock's price series into a few band-limited modes (trend vs. noise). VMD hyperparameters are tuned per stock with a genetic algorithm.
2. **Three relational graphs** (Pearson, Spearman, Dynamic Time Warping) capture how stocks move together.
3. A **Graph-Convolutional GRU (`gcgru`)** learns a weighted mix of the three graphs and predicts the next-day close for each mode.
4. The per-mode predictions are **summed back together** to get the final price.
5. A custom loss (MSE plus a trend-direction penalty) optimizes directly for **trend accuracy**.

| Model (DJIA) | Trend accuracy |
|---|---|
| REGCN baseline | 0.752 |
| Refined REGCN (this project) | 0.793 |

About 200 trained models are converted to **ONNX**, so the production backend serves them with `onnxruntime` and never imports TensorFlow.

## Architecture

```
React + Vite (Tailwind)
        |  /api  (Vite proxy in dev)
        v
FastAPI backend  ----  yfinance (live OHLCV)
  |-- /predict, /predictions  -> VMD + ONNX Runtime inference (1 h cache)
  |-- /news/{query}           -> Alpha Vantage -> NewsAPI -> DuckDuckGo
  |-- /chat (streaming)       -> LangGraph ReAct agent on Groq (Llama 3.1)
  |-- /auth/*                 -> argon2-hashed passwords
  `-- Postgres (prod) / SQLite (local): users, threads, agent checkpoints
```

The chat assistant calls the same `predict_stock()` function as the REST endpoint, so the chat and the dashboard always agree.

## Tech stack

| Layer | Tools |
|---|---|
| Frontend | React, Vite, Tailwind CSS |
| Backend | Python, FastAPI, Uvicorn |
| ML | TensorFlow/Keras (training), ONNX Runtime (serving), VMD, graph neural networks |
| Agent | LangGraph, Groq (Llama 3.1 8B) |
| Data | yfinance, Alpha Vantage, NewsAPI, DuckDuckGo |
| Storage | Postgres (production), SQLite (local fallback) |
| Deploy | Vercel (frontend), Render (backend) |

## Getting started

### Prerequisites

- Node.js 18+ and npm
- Python 3.12+

### Install

```bash
# frontend
npm install

# backend
cd backend_fastapi
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux
pip install -r requirements.txt
cd ..
```

### Environment variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_key
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key
NEWS_API_KEY=your_news_api_key

# Optional: use Postgres instead of the local SQLite file
DATABASE_URL=
```

### Run locally

```bash
npm run dev
```

- Frontend: http://localhost:8080
- FastAPI backend: http://localhost:9000 (interactive docs at `/docs`)

### Quick checks

- `GET http://localhost:9000/api/predict/AAPL` returns a next-day prediction
- `GET http://localhost:9000/api/news/market` returns market news

## Project structure

```
src/                React app (pages, components, hooks, contexts)
backend_fastapi/    FastAPI app, ONNX inference, LangGraph agent, auth
public/             Static assets
```

## Scripts

| Command | Description |
|---|---|
| `npm run dev` | Start the frontend and FastAPI backend |
| `npm run build` | Production build of the React app |
| `npm run preview` | Preview the production build |
| `npm run lint` | Run ESLint |

## Known limitations and roadmap

- **Auth:** the session token is a placeholder, not a signed JWT, and Google sign-in does not yet verify the ID token server-side. Next step: signed, expiring JWTs plus route-level verification.
- **Confidence score:** a heuristic based on agreement between VMD modes, not a calibrated probability. Next step: calibration or conformal prediction intervals.
- **Caching:** prediction and news caches are in-process. Next step: Redis for multi-instance deployments.
- **Inference concurrency:** predictions run behind a single global lock. Next step: per-model locking or a worker pool.
- **Disclaimer:** MarketMind is a research and educational project. Predictions are not financial advice.

## Author

Himanshu Jain, B.Tech Computer Engineering, NIT Kurukshetra
GitHub: [HimanshuJain-2004](https://github.com/HimanshuJain-2004)
