# Hosting Guide — IU Transit Tracker

## Recommended stack: Railway (backend) + Vercel (frontend)
## Total cost: ~$5/month after Railway's free $5 credit runs out

---

## Backend → Railway

Railway runs your FastAPI server 24/7, supports WebSockets, and
gives you $5 free credit (roughly 1 free month).

### Steps

1. Go to https://railway.app and sign up with GitHub

2. In the backend/ folder, create these two files:

**Procfile** (tells Railway how to start the server):
```
web: uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

**railway.toml** (optional but helpful):
```toml
[build]
builder = "nixpacks"

[deploy]
startCommand = "uvicorn app.main:app --host 0.0.0.0 --port $PORT"
healthcheckPath = "/api/health"
```

3. From the Railway dashboard:
   - New Project → Deploy from GitHub repo
   - Set root directory to `backend/`
   - Add environment variables (same as your .env):
     ```
     GTFS_STATIC_URL=https://s3.amazonaws.com/...
     GTFS_RT_VEHICLE_URL=https://s3.amazonaws.com/...
     GROQ_API_KEY=gsk_...
     MAPBOX_TOKEN=pk...
     DATABASE_URL=sqlite+aiosqlite:///./data/transit.db
     DEBUG=false
     ```

4. Railway gives you a URL like `iu-transit.railway.app`

### Persistent database on Railway
SQLite data is wiped on redeploy. For persistence, either:
- Add Railway's Postgres addon ($5/month) and change DATABASE_URL
- Or mount a Railway Volume (persistent disk, free up to 1GB)
  In Railway dashboard: Service → Volumes → Add Volume → mount at `/app/data`

---

## Frontend → Vercel (free forever)

1. Go to https://vercel.com and sign up with GitHub

2. In frontend/.env.local, update the API URL to point to Railway:
```
VITE_MAPBOX_TOKEN=pk.your_token
VITE_API_URL=https://your-app.railway.app
```

3. Update frontend/vite.config.js to use the env var:
```js
server: {
  proxy: {
    '/api': {
      target: process.env.VITE_API_URL || 'http://localhost:8000',
      changeOrigin: true,
    }
  }
}
```

4. From Vercel dashboard:
   - New Project → Import from GitHub
   - Set root directory to `frontend/`
   - Add environment variables:
     ```
     VITE_MAPBOX_TOKEN=pk.your_token
     VITE_API_URL=https://your-app.railway.app
     ```
   - Deploy

5. Vercel gives you `iu-transit.vercel.app` (or custom domain)

---

## WebSocket on Vercel

Vercel doesn't support WebSockets for the frontend proxy.
In production, the frontend must connect directly to the Railway backend URL.

Update frontend/src/store/transitStore.js connectWebSocket:
```js
connectWebSocket: () => {
  const backendUrl = import.meta.env.VITE_API_URL || ''
  const wsUrl = backendUrl
    ? backendUrl.replace('https://', 'wss://').replace('http://', 'ws://') + '/api/buses/live'
    : `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/api/buses/live`
  const ws = new WebSocket(wsUrl)
  // ... rest of handler
}
```

---

## Cost summary

| Service  | Cost         | What it runs              |
|----------|-------------|---------------------------|
| Railway  | $5/mo        | FastAPI, SQLite, scheduler, WebSocket |
| Vercel   | Free         | React frontend            |
| Mapbox   | Free (50k loads/mo) | Satellite map       |
| Groq     | Free (500 req/day) | Llama 3.3 70B LLM    |
| Open-Meteo | Free       | Weather API               |

**Total: ~$5/month** (or free for first month on Railway)

For a student project, Railway's $5 credit covers the first month entirely.
After that, the $5/month Hobby plan keeps it running indefinitely.

---

## Alternative: Fly.io (slightly more setup, generous free tier)

Fly.io gives 3 shared VMs free with no credit card and never spins down.
Better than Railway's free tier but more complex to set up.

```bash
# Install flyctl
brew install flyctl  # or: curl -L https://fly.io/install.sh | sh

# From backend/ directory
fly launch
fly secrets set GROQ_API_KEY=gsk_... MAPBOX_TOKEN=pk...
fly deploy
```
