// src/components/StatusBar.jsx
// Slim bottom bar showing model status, last poll time, and training progress.
// Collapses to a single line on mobile.

import { useState, useEffect } from 'react'
import { useTransitStore } from '../store/transitStore'

export default function StatusBar() {
  const { modelStatus, vehicles, wsConnected } = useTransitStore()
  const [lastUpdate, setLastUpdate] = useState(null)

  // Track when vehicles last updated
  useEffect(() => {
    if (Object.keys(vehicles).length > 0) {
      setLastUpdate(new Date())
    }
  }, [vehicles])

  const modelLoaded = modelStatus?.model_loaded ?? false
  const mode = modelStatus?.mode ?? 'loading'
  const trainingSamples = modelStatus?.training_samples ?? 0

  function timeAgo(date) {
    if (!date) return 'never'
    const s = Math.round((Date.now() - date.getTime()) / 1000)
    if (s < 5) return 'just now'
    if (s < 60) return `${s}s ago`
    return `${Math.round(s / 60)}m ago`
  }

  const [tick, setTick] = useState(0)
  useEffect(() => {
    const id = setInterval(() => setTick((t) => t + 1), 5000)
    return () => clearInterval(id)
  }, [])

  return (
    <footer className="statusbar">
      {/* Model mode pill */}
      <div className={`statusbar__mode ${modelLoaded ? 'statusbar__mode--lstm' : 'statusbar__mode--heuristic'}`}>
        {modelLoaded ? '🧠 LSTM' : '📐 Heuristic'}
      </div>

      {/* Last update */}
      <span className="statusbar__text">
        Updated {timeAgo(lastUpdate)}
      </span>

      {/* WS connection */}
      <span className="statusbar__text">
        {wsConnected ? '● Live' : '○ Reconnecting'}
      </span>

      {/* Training progress (shown when model not loaded) */}
      {!modelLoaded && (
        <span className="statusbar__text statusbar__training">
          Collecting training data…
        </span>
      )}
    </footer>
  )
}
