# IU Transit Tracker

Real-time Bloomington Transit bus tracking with LSTM-based delay prediction that accounts for **IU class release schedules** and **weather conditions** — two signals the official ETAspot app ignores.

---

## Why this exists

ETAspot shows you where buses are. It does not tell you that the 4E at Sample Gates will be 8 minutes late because SPEA's 200-person lecture just ended and 150 students are walking to the stop in the rain.

This app does.

---

## Architecture

```
GTFS-RT (15s)  ──┐
IU Schedule    ──┼──► Python Ingest ──► SQLite ──► LSTM ──► FastAPI ──► React (Mapbox satellite)
Open-Meteo(5m) ──┘                                      └──► WebSocket ──► live bus dots
```

**Backend**: FastAPI + SQLAlchemy async + APScheduler  
**ML**: PyTorch LSTM (2-layer, 64 hidden, Gaussian NLL loss)  
**Frontend**: React + react-map-gl + Mapbox satellite-streets + Zustand  

---

## LSTM rationale

We use an LSTM (not a Transformer, not LLaMA) because:

1. **Short sequences** — 10 timesteps × 15s = 2.5 min of history. Transformers only beat LSTMs at 100+ steps.
2. **Small initial dataset** — LSTMs generalise from <1,000 samples. Transformers need orders of magnitude more.
3. **Causal temporal structure** — delay at stop A propagates forward. LSTM hidden state models this naturally.
4. **Inference speed** — microseconds per sample on CPU. Runs fine on a $5/mo VPS.

Once 30+ days of data accumulate (~200k samples), we can revisit N-BEATS or a small Transformer variant if LSTM error plateaus.

---

## Quick start

### Prerequisites
- Python 3.11+
- Node.js 18+
- A free [Mapbox account](https://account.mapbox.com/) for the satellite map

### 1. Backend

```bash
cd backend
bash setup.sh          # creates venv, installs deps, probes GTFS-RT endpoints
cp .env.example .env   # edit: set MAPBOX_TOKEN + correct GTFS_RT_VEHICLE_URL
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload
```

### 2. Upload the IU class schedule

```bash
# Upload your exported IU schedule CSV (the 8,337-row file)
curl -X POST http://localhost:8000/api/admin/load-schedule \
     -F 'file=@/path/to/schedule.csv'
```

This automatically:
- Parses all in-person sections
- Geocodes every building code to lat/lng
- Computes the student-release event cache for all stops

### 3. Frontend

```bash
cd frontend
cp .env.example .env.local   # set VITE_MAPBOX_TOKEN
npm install
npm run dev                   # opens http://localhost:3000
```

### 4. Find the correct GTFS-RT URL

```bash
cd backend
source .venv/bin/activate
python scripts/probe_gtfs_rt.py
```

Update `GTFS_RT_VEHICLE_URL` in `.env` with the working URL, then restart the server.

---

## API reference

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/api/buses/routes` | All routes with colours |
| `GET`  | `/api/buses/stops` | All stops with coordinates |
| `GET`  | `/api/buses/shapes/{route_id}` | Route polyline |
| `GET`  | `/api/buses/vehicles` | Current positions (HTTP) |
| `WS`   | `/api/buses/live` | Live position stream |
| `GET`  | `/api/buses/stop/{id}/schedule` | Upcoming arrivals |
| `GET`  | `/api/predictions/stop/{id}` | LSTM prediction |
| `GET`  | `/api/predictions/status` | Model mode + weather |
| `POST` | `/api/admin/load-schedule` | Upload class schedule CSV |
| `GET`  | `/api/admin/status` | System health + DB counts |
| `GET`  | `/api/admin/probe-gtfs-rt` | Test RT endpoint candidates |
| `POST` | `/api/admin/train-model` | Trigger LSTM training |
| `GET`  | `/api/docs` | Swagger UI |

---

## Feature vector (LSTM input, 9 features per timestep)

| Index | Feature | Encoding |
|-------|---------|----------|
| 0–1 | Hour of day | sin/cos cyclical |
| 2–3 | Minute | sin/cos cyclical |
| 4–5 | Day of week | sin/cos cyclical |
| 6 | Current delay (seconds) | z-scored, clipped ±3σ |
| 7 | Weather severity | 0.0 (clear) – 1.0 (severe) |
| 8 | Students releasing nearby | normalised 0–1, 400m radius, 15min lookahead |

---

## Day-one operation guide

**Hour 0–1**: The LSTM has no training data. The system uses a heuristic fallback — it estimates delays based on weather severity and student-release counts. The status bar shows "📐 Heuristic". This is still better than ETAspot because it accounts for class schedules.

**Hour 2–4**: ~1,000 training samples accumulate. Trigger training manually:
```bash
curl -X POST http://localhost:8000/api/admin/train-model
```

**After 1 day**: The scheduler auto-retrains every 4 hours. The status bar switches to "🧠 LSTM". Predictions improve as the model learns your specific routes' delay patterns.

**After 1 week**: Enough data to see meaningful signal from the student-release feature. You'll see the model correctly predicting delays at stops near Ballantine, SPEA, and Luddy at class-break times.

---

## Planned improvements

- [ ] **Syllabus scraper** — scrape `syllabi.iu.edu` to estimate actual attendance probability based on whether attendance is graded, when exams fall, and how far into the semester we are
- [ ] **Stop-to-building proximity refinement** — weight student counts by walking time, not just radius
- [ ] **Game day signal** — IU football/basketball events create massive transit demand spikes
- [ ] **Driver-behaviour features** — some drivers are consistently 2 min early/late; encode vehicle_id as a learnable embedding
- [ ] **Multi-step prediction** — predict delay at stop N+3, not just N+1

---

## Project structure

```
iu-transit/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI app + startup
│   │   ├── core/
│   │   │   ├── config.py        # Settings (pydantic-settings)
│   │   │   ├── database.py      # Async SQLAlchemy
│   │   │   ├── scheduler.py     # APScheduler jobs
│   │   │   └── websocket.py     # WS connection manager
│   │   ├── models/
│   │   │   ├── gtfs.py          # Routes, stops, trips, vehicle positions
│   │   │   ├── schedule.py      # IU class sections, release events
│   │   │   ├── weather.py       # Weather observations
│   │   │   └── predictions.py   # LSTM prediction log
│   │   ├── services/
│   │   │   ├── gtfs_static.py   # Download + parse GTFS zip
│   │   │   ├── gtfs_realtime.py # Poll GTFS-RT protobuf
│   │   │   ├── weather.py       # Open-Meteo fetcher
│   │   │   ├── class_schedule.py# CSV parser + release events
│   │   │   └── geocoder.py      # Building code → lat/lng
│   │   ├── ml/
│   │   │   ├── lstm_model.py    # PyTorch LSTM + predictor
│   │   │   ├── trainer.py       # Training loop + dataset builder
│   │   │   └── feature_builder.py # Per-stop feature assembly
│   │   └── api/
│   │       ├── buses.py         # Routes/stops/vehicles/WS endpoints
│   │       ├── predictions.py   # LSTM prediction endpoints
│   │       └── admin.py         # Data loading + system status
│   ├── scripts/
│   │   └── probe_gtfs_rt.py     # Identify correct RT endpoint
│   ├── requirements.txt
│   ├── setup.sh
│   └── .env.example
└── frontend/
    ├── src/
    │   ├── main.jsx
    │   ├── App.jsx
    │   ├── store/
    │   │   └── transitStore.js  # Zustand global state + WS client
    │   ├── components/
    │   │   ├── MapView.jsx      # Mapbox satellite map
    │   │   ├── BusMarker.jsx    # Animated bus dot
    │   │   ├── StopMarker.jsx   # Tappable stop dot
    │   │   ├── StopPanel.jsx    # Slide-up prediction panel
    │   │   ├── TopBar.jsx       # Route filters + WS status
    │   │   └── StatusBar.jsx    # Model mode + last update
    │   └── styles/
    │       └── global.css       # Mobile-first IU theme
    ├── index.html
    ├── vite.config.js
    ├── package.json
    └── .env.example
```
