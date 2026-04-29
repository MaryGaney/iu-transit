// src/components/VehiclePopup.jsx
import { useTransitStore } from '../store/transitStore'

const OCC_CFG = {
  empty:     { label: 'Empty',               emoji: '🟢', pct: 5,   color: '#22C55E' },
  low:       { label: 'Seats available',      emoji: '🟢', pct: 25,  color: '#22C55E' },
  moderate:  { label: 'Filling up',           emoji: '🟡', pct: 55,  color: '#F59E0B' },
  high:      { label: 'Nearly full',          emoji: '🟠', pct: 80,  color: '#F97316' },
  very_high: { label: 'Standing room only',   emoji: '🔴', pct: 100, color: '#EF4444' },
  crush:     { label: 'Standing room only',   emoji: '🔴', pct: 100, color: '#EF4444' },
  unknown:   { label: 'Estimating…',          emoji: '⚪', pct: 0,   color: '#9CA3AF' },
}

export default function VehiclePopup() {
  const { selectedVehicle, vehicleOccupancy, clearSelectedVehicle, routes } = useTransitStore()
  if (!selectedVehicle) return null

  const s = (v) => (v != null ? String(v).trim() : '')
  const vehicleRouteId = s(selectedVehicle.route_id)
  const route   = routes.find(r => s(r.route_id) === vehicleRouteId)
    || routes.find(r => s(r.short_name) === vehicleRouteId)
  const color   = route?.color || '#2563EB'
  const simLabel = s(selectedVehicle.vehicle_id).startsWith('SIM-')
    ? s(selectedVehicle.vehicle_id).split('-')[1] : ''
  const label   =
    s(route?.short_name) ||
    s(route?.route_id) ||
    vehicleRouteId ||
    simLabel ||
    '?'

  const occ        = vehicleOccupancy || {}
  const cfg        = OCC_CFG[occ.occupancy_level] || OCC_CFG.unknown
  const delayProb  = occ.delay_probability ?? 0
  const factors    = occ.factors || {}
  const releasing  = factors.class_release   || {}
  const starting   = factors.class_starting  || {}
  const weatherF   = factors.weather         || {}
  const tempF      = factors.temperature     || {}

  const delayColor = delayProb > 0.6 ? '#EF4444' : delayProb > 0.3 ? '#F59E0B' : '#22C55E'

  return (
    <div className="vehicle-popup">
      {/* Header */}
      <div className="vp-header">
        <span className="vp-badge" style={{ background: color }}>{label}</span>
        <span className="vp-name">{route?.long_name || `Route ${label}`}</span>
        {selectedVehicle.simulated && <span className="vp-sim">SIM</span>}
        <button className="vp-close" onClick={clearSelectedVehicle}>✕</button>
      </div>

      {/* Occupancy */}
      <div className="vp-block">
        <div className="vp-label">Estimated occupancy</div>
        <div className="vp-row">
          <span style={{ fontSize: 20 }}>{cfg.emoji}</span>
          <div style={{ flex: 1 }}>
            <div className="vp-value">{cfg.label}</div>
            <div className="vp-bar"><div className="vp-bar__fill" style={{ width: `${cfg.pct}%`, background: cfg.color }} /></div>
          </div>
        </div>
      </div>

      {/* Delay likelihood */}
      <div className="vp-block">
        <div className="vp-label">Delay likelihood</div>
        <div className="vp-row" style={{ gap: 8 }}>
          <div className="vp-bar" style={{ flex: 1 }}>
            <div className="vp-bar__fill" style={{ width: `${Math.round(delayProb * 100)}%`, background: delayColor }} />
          </div>
          <span className="vp-value">{Math.round(delayProb * 100)}%</span>
        </div>
      </div>

      {/* Factors */}
      <div className="vp-block">
        <div className="vp-label">What's driving this</div>
        <div className="vp-factors">

          <div className={`vp-factor ${releasing.active ? 'vp-factor--on' : ''}`}>
            <span>🎓</span>
            <span className="vp-factor__text">{releasing.label || 'No classes releasing soon'}</span>
          </div>

          <div className={`vp-factor ${starting.active ? 'vp-factor--on' : ''}`}>
            <span>📚</span>
            <span className="vp-factor__text">{starting.label || 'No classes starting soon'}</span>
          </div>

          <div className={`vp-factor ${weatherF.active ? 'vp-factor--on' : ''}`}>
            <span>{weatherF.is_snowing ? '❄️' : weatherF.is_raining ? '🌧' : '☀️'}</span>
            <span className="vp-factor__text">{weatherF.label || 'Clear weather'}</span>
          </div>

          <div className={`vp-factor ${tempF.active ? 'vp-factor--on' : ''}`}>
            <span>
              {(tempF.temp_f ?? 70) >= 90 ? '🥵' : (tempF.temp_f ?? 70) <= 25 ? '🥶' : '🌡'}
            </span>
            <span className="vp-factor__text">{tempF.label || 'Normal temperature'}</span>
          </div>

        </div>
      </div>

      <div className="vp-footer">
        📐 Heuristic · improves as LSTM trains
      </div>
    </div>
  )
}
