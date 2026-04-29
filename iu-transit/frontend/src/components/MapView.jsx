// src/components/MapView.jsx
import { useRef, useCallback } from 'react'
import Map, { Source, Layer, NavigationControl } from 'react-map-gl'
import { useTransitStore } from '../store/transitStore'
import BusMarker from './BusMarker'
import StopMarker from './StopMarker'
import VehiclePopup from './VehiclePopup'
import 'mapbox-gl/dist/mapbox-gl.css'

const INITIAL_VIEW = {
  latitude: 39.1686,
  longitude: -86.5254,
  zoom: 14.5,
  bearing: 0,
  pitch: 0,
}

export default function MapView() {
  // Subscribe to each piece of state individually so Zustand
  // re-renders this component when any of them change
  const vehicles        = useTransitStore(s => s.vehicles)
  const stops           = useTransitStore(s => s.stops)
  const routes          = useTransitStore(s => s.routes)
  const routeShapes     = useTransitStore(s => s.routeShapes)
  const showRoutes      = useTransitStore(s => s.showRoutes)
  const showStops       = useTransitStore(s => s.showStops)
  const showHeatmap     = useTransitStore(s => s.showHeatmap)
  const heatmapData     = useTransitStore(s => s.heatmapData)
  const activeRoutesMap = useTransitStore(s => s.activeRoutesMap)
  const selectStop      = useTransitStore(s => s.selectStop)
  const selectVehicle   = useTransitStore(s => s.selectVehicle)
  const selectedVehicle = useTransitStore(s => s.selectedVehicle)
  const setMapLoaded    = useTransitStore(s => s.setMapLoaded)

  const handleMapLoad = useCallback(() => setMapLoaded?.(true), [])
  const mapboxToken = import.meta.env.VITE_MAPBOX_TOKEN || ''

  // Empty object = show all routes; otherwise only those explicitly selected
  const noneSelected = Object.keys(activeRoutesMap).length === 0
  const isVisible = (route_id) => noneSelected || activeRoutesMap[route_id] === true

  const allVehicles = Object.values(vehicles)
  const visibleVehicles = noneSelected
    ? allVehicles
    : allVehicles.filter(v => activeRoutesMap[v.route_id] === true)

  const heatGeoJSON = heatmapData && heatmapData.length > 0 ? {
    type: 'FeatureCollection',
    features: heatmapData.map(pt => ({
      type: 'Feature',
      geometry: { type: 'Point', coordinates: [pt.lng, pt.lat] },
      properties: { weight: pt.weight },
    })),
  } : null

  return (
    <Map
      mapboxAccessToken={mapboxToken}
      initialViewState={INITIAL_VIEW}
      style={{ width: '100%', height: '100%' }}
      mapStyle="mapbox://styles/mapbox/satellite-streets-v12"
      onLoad={handleMapLoad}
      maxZoom={18}
      minZoom={11}
    >
      <NavigationControl position="top-right" />

      {/* ── Route polylines ── */}
      {showRoutes && routes.map((route) => {
        if (!isVisible(route.route_id)) return null
        const shapeData = routeShapes[route.route_id]
        if (!shapeData) return null

        const color = route.color || '#2563EB'

        const polylines =
          Array.isArray(shapeData.shapes) && shapeData.shapes.length > 0
            ? shapeData.shapes.map(s => s.points)
            : Array.isArray(shapeData.points) && shapeData.points.length > 0
              ? [shapeData.points]
              : []

        return polylines.map((pts, idx) => {
          if (!pts || pts.length < 2) return null
          const sid = `s-${route.route_id}-${idx}`
          const geojson = {
            type: 'Feature',
            geometry: {
              type: 'LineString',
              coordinates: pts.map(p => [p.lng, p.lat]),
            },
          }
          return (
            <Source key={sid} id={sid} type="geojson" data={geojson}>
              <Layer id={`${sid}-halo`} type="line"
                paint={{ 'line-color': '#fff', 'line-width': 8, 'line-opacity': 0.55 }}
                layout={{ 'line-join': 'round', 'line-cap': 'round' }} />
              <Layer id={`${sid}-line`} type="line"
                paint={{ 'line-color': color, 'line-width': 5, 'line-opacity': 1 }}
                layout={{ 'line-join': 'round', 'line-cap': 'round' }} />
              {idx === 0 && (
                <Layer id={`${sid}-lbl`} type="symbol"
                  layout={{
                    'symbol-placement': 'line',
                    'text-field': route.short_name || route.route_id,
                    'text-size': 12,
                    'text-font': ['DIN Offc Pro Bold', 'Arial Unicode MS Bold'],
                    'symbol-spacing': 300,
                    'text-keep-upright': true,
                  }}
                  paint={{
                    'text-color': '#fff',
                    'text-halo-color': color,
                    'text-halo-width': 2.5,
                  }} />
              )}
            </Source>
          )
        })
      })}

      {/* ── Heatmap ── */}
      {showHeatmap && heatGeoJSON && (
        <Source id="heat-src" type="geojson" data={heatGeoJSON}>
          <Layer id="heat-layer" type="heatmap"
            paint={{
              'heatmap-weight': ['get', 'weight'],
              'heatmap-intensity': 2.5,
              'heatmap-radius': [
                'interpolate', ['linear'], ['zoom'],
                11, 30,
                14, 60,
                16, 90,
              ],
              'heatmap-opacity': 0.85,
              // Pure red-dominant palette — clearly visible on satellite
              'heatmap-color': [
                'interpolate', ['linear'], ['heatmap-density'],
                0,    'rgba(0,0,0,0)',
                0.05, 'rgba(255,255,0,0.4)',
                0.25, 'rgba(255,165,0,0.7)',
                0.5,  'rgba(255,60,0,0.85)',
                0.75, 'rgba(220,0,0,0.95)',
                1,    'rgba(153,0,0,1)',
              ],
            }} />
        </Source>
      )}

      {/* ── Stops ── */}
      {showStops && stops.map(stop => (
        <StopMarker key={stop.stop_id} stop={stop} onClick={() => selectStop(stop)} />
      ))}

      {/* ── Buses ── */}
      {visibleVehicles.map(vehicle => (
        <BusMarker
          key={vehicle.vehicle_id}
          vehicle={vehicle}
          routes={routes}
          onClick={() => selectVehicle(vehicle)}
          isSelected={selectedVehicle?.vehicle_id === vehicle.vehicle_id}
        />
      ))}

      {selectedVehicle && <VehiclePopup />}
    </Map>
  )
}
