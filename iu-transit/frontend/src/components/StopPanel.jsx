// src/components/StopPanel.jsx
import { useTransitStore } from '../store/transitStore'

const REASON_CONFIG = {
  on_time:        { label: 'On time',          color: '#22C55E', bg: '#DCFCE7' },
  class_release:  { label: 'Class letting out', color: '#F59E0B', bg: '#FEF3C7' },
  class_starting: { label: 'Class starting',   color: '#8B5CF6', bg: '#EDE9FE' },
  weather:        { label: 'Weather delay',     color: '#3B82F6', bg: '#DBEAFE' },
  cascading:      { label: 'Cascading delay',   color: '#EF4444', bg: '#FEE2E2' },
  unknown:        { label: 'Possible delay',    color: '#6B7280', bg: '#F3F4F6' },
}

function formatDelay(seconds) {
  if (seconds == null) return null
  const mins = Math.round(Math.abs(seconds) / 60)
  if (mins === 0) return 'On time'
  return `${seconds > 0 ? '+' : '-'}${mins} min`
}

function formatTime(timeStr) {
  if (!timeStr) return ''
  const [h, m] = timeStr.split(':').map(Number)
  const period = h >= 12 ? 'PM' : 'AM'
  return `${h % 12 || 12}:${String(m).padStart(2, '0')} ${period}`
}

function DelayBadge({ reason, delaySeconds }) {
  const cfg = REASON_CONFIG[reason] || REASON_CONFIG.unknown
  return (
    <span className="delay-badge" style={{ color: cfg.color, background: cfg.bg }}>
      {cfg.label}{delaySeconds > 30 ? ` · ${formatDelay(delaySeconds)}` : ''}
    </span>
  )
}

function ConfidenceRing({ confidence }) {
  const pct    = Math.round((confidence || 0) * 100)
  const r      = 16
  const circ   = 2 * Math.PI * r
  const dash   = (circ * pct) / 100
  return (
    <div className="confidence-ring" title={`${pct}% confidence`}>
      <svg width="40" height="40" viewBox="0 0 40 40">
        <circle cx="20" cy="20" r={r} fill="none" stroke="#E5E7EB" strokeWidth="3" />
        <circle cx="20" cy="20" r={r} fill="none" stroke="#3B82F6" strokeWidth="3"
          strokeDasharray={`${dash} ${circ}`} strokeLinecap="round"
          transform="rotate(-90 20 20)" />
      </svg>
      <span className="confidence-ring__label">{pct}%</span>
    </div>
  )
}

export default function StopPanel() {
  const { selectedStop, stopSchedule, stopPrediction, clearSelectedStop } = useTransitStore()
  if (!selectedStop) return null

  const arrivals     = stopSchedule?.arrivals   || []
  const predictions  = stopPrediction?.predictions || []
  const topPred      = predictions[0]

  return (
    <div className="stop-panel">
      <div className="stop-panel__handle" />

      <div className="stop-panel__header">
        <div>
          <h2 className="stop-panel__name">{selectedStop.name}</h2>
          <p className="stop-panel__id">Stop #{selectedStop.stop_id}</p>
        </div>
        <button className="stop-panel__close" onClick={clearSelectedStop}>✕</button>
      </div>

      {topPred && (
        <div className="prediction-banner">
          <div className="prediction-banner__left">
            <div className="prediction-banner__title">
              {topPred.model_used ? 'LSTM prediction' : 'Estimated'}
            </div>
            <DelayBadge reason={topPred.delay_reason} delaySeconds={topPred.predicted_delay_seconds} />
            <div className="prediction-banner__range">
              Range: {formatDelay(topPred.predicted_delay_lower)} to {formatDelay(topPred.predicted_delay_upper)}
            </div>
          </div>
          <ConfidenceRing confidence={topPred.confidence} />
        </div>
      )}

      <div className="stop-panel__section">
        <h3 className="stop-panel__section-title">Upcoming arrivals</h3>
        {arrivals.length === 0
          ? <p className="stop-panel__empty">No scheduled arrivals in the next hour.</p>
          : (
            <div className="arrivals-list">
              {arrivals.map((arr, i) => {
                const pred     = predictions.find(p => p.route_id === arr.route_id)
                const delaySec = pred?.predicted_delay_seconds ?? 0
                return (
                  <div className="arrival-row" key={i}>
                    <span className="arrival-row__badge"
                      style={{ background: arr.route_color || '#3B82F6' }}>
                      {arr.route_short_name}
                    </span>
                    <div className="arrival-row__info">
                      <span className="arrival-row__headsign">
                        {arr.headsign || arr.route_long_name}
                      </span>
                      <span className="arrival-row__time">{formatTime(arr.arrival_time)}</span>
                    </div>
                    {delaySec > 30 && (
                      <span className="arrival-row__delay" style={{ color: '#EF4444' }}>
                        {formatDelay(delaySec)}
                      </span>
                    )}
                  </div>
                )
              })}
            </div>
          )
        }
      </div>

      {topPred && !topPred.model_used && (
        <p className="stop-panel__model-note">
          ⓘ LSTM training in progress — showing heuristic estimate
        </p>
      )}
    </div>
  )
}
