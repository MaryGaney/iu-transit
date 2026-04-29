// src/components/StopMarker.jsx
import { Marker } from 'react-map-gl'
import { useTransitStore } from '../store/transitStore'

export default function StopMarker({ stop, onClick }) {
  const selectedStop = useTransitStore((s) => s.selectedStop)
  const isSelected = selectedStop?.stop_id === stop.stop_id

  return (
    <Marker
      longitude={stop.lng}
      latitude={stop.lat}
      anchor="center"
      onClick={(e) => {
        e.originalEvent.stopPropagation()
        onClick()
      }}
    >
      <div
        className={`stop-marker ${isSelected ? 'stop-marker--selected' : ''}`}
        title={stop.name}
      />
    </Marker>
  )
}
