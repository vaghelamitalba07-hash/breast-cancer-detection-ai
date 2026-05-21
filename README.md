# Breast Cancer Detection AI — Research Prototype

**Not a medical device.** Research and education only.

## Project structure

```
breast-cancer-detection-ai-main/
├── frontend/          # Web UI (HTML, CSS, JavaScript)
│   ├── index.html     # Landing page
│   ├── login.html
│   ├── signup.html
│   ├── dashboard.html
│   ├── css/
│   └── js/
├── backend/           # API + ML
│   ├── main.py        # FastAPI server (API + serves frontend)
│   ├── ml_service.py  # Model train/load/predict
│   ├── auth_service.py
│   ├── config.py
│   ├── data/users.json
│   └── requirements.txt
└── README.md
```

## Stack (minimal — only what the web app needs)

| Layer | Technology |
|-------|------------|
| Frontend | HTML, CSS, vanilla JavaScript |
| Backend | FastAPI, Uvicorn |
| ML | scikit-learn, pandas, joblib |

No React, Streamlit, database, or Docker (unless you add later).

## Run locally

```bash
cd backend
py -3.13 -m pip install -r requirements.txt
py -3.13 -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Open: **http://localhost:8000**

## Demo login

- Username: `demo` — Password: `demo123`
- Username: `researcher` — Password: `research2024`

## API endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/api/health` | No | Health check |
| POST | `/api/auth/login` | No | Sign in |
| POST | `/api/auth/register` | No | Sign up |
| POST | `/api/auth/logout` | Yes | Sign out |
| GET | `/api/metrics` | Yes | Model metrics |
| POST | `/api/predict` | Yes | Run prediction |

## Disclaimer

Not FDA-approved. Do not use for clinical decisions. Always consult qualified radiologists.
