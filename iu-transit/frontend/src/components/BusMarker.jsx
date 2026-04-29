// src/components/BusMarker.jsx
import { Marker } from 'react-map-gl'

const OCCUPANCY_COLORS = {
  empty:     '#22C55E',
  low:       '#22C55E',
  moderate:  '#F59E0B',
  high:      '#F97316',
  very_high: '#EF4444',
  crush:     '#991B1B',
}

// Coerce any value to a trimmed string — never throws, never returns null/undefined
const s = (v) => (v != null ? String(v).trim() : '')

export default function BusMarker({ vehicle, routes, onClick, isSelected }) {
  // Coerce route_id to string — GTFS can sometimes send integers
  const vehicleRouteId = s(vehicle.route_id)

  // Match by route_id (exact), then by short_name
  const route = routes.find(r => s(r.route_id) === vehicleRouteId)
    || routes.find(r => s(r.short_name) === vehicleRouteId)

  const routeColor = route?.color || '#2563EB'

  // Label priority (all coerced to safe strings):
  //   1. DB short_name (e.g. "1", "9L", "2S")
  //   2. DB route_id
  //   3. RT feed route_id on the vehicle
  //   4. Sim vehicle_id parse: "SIM-4E-2" → "4E"
  //   5. '?' only if everything above is empty
  const simLabel = s(vehicle.vehicle_id).startsWith('SIM-')
    ? s(vehicle.vehicle_id).split('-')[1]
    : ''

  const label =
    s(route?.short_name) ||
    s(route?.route_id) ||
    vehicleRouteId ||
    simLabel ||
    '?'

  const bearing = typeof vehicle.bearing === 'number' ? vehicle.bearing : 0
  const occColor = OCCUPANCY_COLORS[vehicle.occupancy_level]
  const ringStyle = occColor
    ? `0 0 0 3px ${occColor}, 0 3px 14px rgba(0,0,0,0.6)`
    : `0 0 0 3px rgba(255,255,255,0.75), 0 3px 14px rgba(0,0,0,0.6)`

  return (
    <Marker longitude={vehicle.lng} latitude={vehicle.lat} anchor="center">
      <div
        className={`bus-marker ${isSelected ? 'bus-marker--selected' : ''}`}
        onClick={e => { e.stopPropagation(); onClick?.() }}
        title={`Route ${label} · #${vehicle.vehicle_id}${vehicle.simulated ? ' (sim)' : ''}`}
      >
        <div className="bus-marker__pulse" style={{ borderColor: routeColor }} />

        <div
          className="bus-marker__body"
          style={{ background: routeColor, boxShadow: ringStyle }}
        >
          <span className="bus-marker__label">{label}</span>
        </div>

        <div
          className="bus-marker__rotator"
          style={{ transform: `rotate(${bearing}deg)` }}
        >
          <div
            className="bus-marker__arrow"
            style={{ '--arrow-color': routeColor }}
          />
        </div>
      </div>
    </Marker>
  )
}
