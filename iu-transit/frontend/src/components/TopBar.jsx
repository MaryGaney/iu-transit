// src/components/TopBar.jsx
import { useState } from 'react'
import { useTransitStore } from '../store/transitStore'

const WMO_ICONS = {
  0:'☀️',1:'🌤',2:'⛅',3:'☁️',
  45:'🌫',48:'🌫',
  51:'🌦',53:'🌧',55:'🌧',61:'🌧',63:'🌧',65:'🌧',
  71:'🌨',73:'❄️',75:'❄️',
  80:'🌦',81:'🌧',82:'⛈',95:'⛈',96:'⛈',99:'⛈',
}

const toF = (c) => Math.round(c * 9/5 + 32)

// Only show simulator in dev (localhost) — never in production
const IS_DEV = window.location.hostname === 'localhost' ||
               window.location.hostname === '127.0.0.1'

export default function TopBar() {
  const {
    routes, wsConnected, vehicles, weather,
    activeRoutesMap, toggleRouteFilter, clearRouteFilter,
    showStops, toggleStops,
    showRoutes, toggleRoutes,
    showHeatmap, toggleHeatmap,
  } = useTransitStore()

  const [simRunning, setSimRunning]   = useState(false)
  const [simLoading, setSimLoading]   = useState(false)

  const vehicleCount = Object.keys(vehicles).length
  const noneSelected = Object.keys(activeRoutesMap).length === 0
  const weatherIcon  = weather ? (WMO_ICONS[weather.weather_code] ?? '🌡') : ''
  const tempF        = weather ? toF(weather.temperature_c) : null
  const weatherAlert = weather && (
    weather.is_raining || weather.is_snowing || weather.is_severe ||
    (tempF !== null && (tempF >= 90 || tempF <= 20))
  )

  const BASE = import.meta.env.VITE_API_URL || ''

  async function toggleSimulator() {
    setSimLoading(true)
    try {
      if (simRunning) {
        await fetch(`${BASE}/api/simulator/stop`, { method: 'POST' })
        setSimRunning(false)
      } else {
        const res  = await fetch(`${BASE}/api/simulator/start?bus_count=10`, { method: 'POST' })
        const data = await res.json()
        if (data.status === 'started' || data.status === 'already_running') {
          setSimRunning(true)
        } else {
          alert('Simulator: ' + (data.message || JSON.stringify(data)))
        }
      }
    } catch (e) {
      alert('Simulator error: ' + e.message)
    } finally {
      setSimLoading(false)
    }
  }

  return (
    <header className="topbar">
      {/* Brand */}
      <div className="topbar__brand">
        <span className="topbar__logo">🚌</span>
        <div>
          <div className="topbar__title">IU Transit</div>
          <div className="topbar__subtitle">Bloomington</div>
        </div>
      </div>

      {/* Live dot */}
      <div className="topbar__live">
        <span className={`topbar__dot ${wsConnected ? 'topbar__dot--live' : 'topbar__dot--offline'}`} />
        <span className="topbar__live-label">
          {wsConnected ? `${vehicleCount} live` : 'connecting…'}
        </span>
      </div>

      {/* Weather */}
      {weather && tempF !== null && (
        <div className={`topbar__weather ${weatherAlert ? 'topbar__weather--alert' : ''}`}>
          {weatherIcon} {tempF}°F
          {weather.is_raining && ' · Rain'}
          {weather.is_snowing && ' · Snow'}
          {weather.is_severe  && ' ⚠️'}
          {tempF >= 90        && ' · Very hot'}
          {tempF <= 20        && ' · Very cold'}
        </div>
      )}

      {/* Layer toggles */}
      <div className="topbar__toggles">
        <button
          className={`topbar__toggle ${showRoutes ? 'topbar__toggle--active' : ''}`}
          onClick={toggleRoutes}
        >Routes</button>
        <button
          className={`topbar__toggle ${showStops ? 'topbar__toggle--active' : ''}`}
          onClick={toggleStops}
        >Stops</button>
        <button
          className={`topbar__toggle ${showHeatmap ? 'topbar__toggle--active topbar__toggle--heat' : ''}`}
          onClick={toggleHeatmap}
          title="Crowding heatmap"
        >🔥 Heat</button>

        {/* Simulator only visible on localhost */}
        {IS_DEV && (
          <button
            className={`sim-btn ${simRunning ? 'sim-btn--active' : ''}`}
            onClick={toggleSimulator}
            disabled={simLoading}
          >
            {simLoading ? '…' : simRunning ? '⏹ Sim' : '▶ Sim'}
          </button>
        )}
      </div>

      {/* Route multi-select chips */}
      {routes.length > 0 && (
        <div className="route-chips">
          <button
            className={`route-chip route-chip--all ${noneSelected ? 'route-chip--all-active' : ''}`}
            onClick={clearRouteFilter}
            title="Show all routes"
          >All</button>

          {routes.map((r) => {
            const active = activeRoutesMap[r.route_id] === true
            return (
              <button
                key={r.route_id}
                className={`route-chip ${active ? 'route-chip--active' : ''}`}
                onClick={() => toggleRouteFilter(r.route_id)}
                style={{ '--chip-color': r.color, '--chip-text': r.text_color || '#fff' }}
                aria-pressed={active}
                title={r.long_name || r.route_id}
              >
                {active && <span className="route-chip__tick">✓ </span>}
                {r.short_name || r.route_id}
              </button>
            )
          })}
        </div>
      )}
    </header>
  )
}
