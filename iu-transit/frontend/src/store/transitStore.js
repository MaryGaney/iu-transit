// src/store/transitStore.js
import { create } from 'zustand'

const API = (import.meta.env.VITE_API_URL || '') + '/api'

// NOTE: Zustand requires NEW object references to trigger re-renders.
// We store activeRoutes as a plain object { route_id: true } NOT a Set,
// because mutating a Set object doesn't trigger Zustand updates.

export const useTransitStore = create((set, get) => ({

  // ── Routes ────────────────────────────────────────────────────────────────
  routes: [],
  routeShapes: {},

  fetchRoutes: async () => {
    try {
      const res = await fetch(`${API}/buses/routes`)
      if (!res.ok) throw new Error(`routes ${res.status}`)
      const routes = await res.json()
      set({ routes })
      // Fetch all shapes in parallel
      const shapes = {}
      await Promise.allSettled(
        routes.map(async (r) => {
          try {
            const s = await fetch(`${API}/buses/shapes/${r.route_id}`)
            if (s.ok) shapes[r.route_id] = await s.json()
          } catch (_) {}
        })
      )
      set({ routeShapes: shapes })
    } catch (e) {
      console.error('fetchRoutes', e)
    }
  },

  // ── Stops ─────────────────────────────────────────────────────────────────
  stops: [],
  fetchStops: async () => {
    try {
      const res = await fetch(`${API}/buses/stops`)
      if (!res.ok) throw new Error(`stops ${res.status}`)
      set({ stops: await res.json() })
    } catch (e) {
      console.error('fetchStops', e)
    }
  },

  // ── Live vehicles via WebSocket ───────────────────────────────────────────
  vehicles: {},
  wsConnected: false,
  wsRef: null,

  connectWebSocket: () => {
    const existing = get().wsRef
    if (existing && existing.readyState <= 1) return

    // In production, connect directly to Railway backend (supports WS)
    // In dev, use Vite proxy (localhost:3000 → localhost:8000)
    const apiUrl = import.meta.env.VITE_API_URL || ''
    const wsUrl = apiUrl
      ? apiUrl.replace('https://', 'wss://').replace('http://', 'ws://') + '/api/buses/live'
      : `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/api/buses/live`

    const ws = new WebSocket(wsUrl)

    ws.onopen = () => set({ wsConnected: true })

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data)
        if (msg.type === 'vehicle_positions') {
          const vehicles = {}
          for (const v of msg.vehicles) vehicles[v.vehicle_id] = v
          set({ vehicles })
        }
      } catch (_) {}
    }

    ws.onclose = () => {
      set({ wsConnected: false, wsRef: null })
      setTimeout(() => get().connectWebSocket(), 3000)
    }
    ws.onerror = () => ws.close()
    set({ wsRef: ws })
  },

  // ── Selected stop ─────────────────────────────────────────────────────────
  selectedStop: null,
  stopSchedule: null,
  stopPrediction: null,

  selectStop: async (stop) => {
    set({ selectedStop: stop, stopSchedule: null, stopPrediction: null, selectedVehicle: null })
    try {
      const [sr, pr] = await Promise.all([
        fetch(`${API}/buses/stop/${stop.stop_id}/schedule`),
        fetch(`${API}/predictions/stop/${stop.stop_id}`),
      ])
      set({ stopSchedule: await sr.json(), stopPrediction: await pr.json() })
    } catch (e) { console.error('selectStop', e) }
  },

  clearSelectedStop: () => set({ selectedStop: null, stopSchedule: null, stopPrediction: null }),

  // ── Selected vehicle ──────────────────────────────────────────────────────
  selectedVehicle: null,
  vehicleOccupancy: null,

  selectVehicle: async (vehicle) => {
    set({ selectedVehicle: vehicle, vehicleOccupancy: null, selectedStop: null })
    try {
      const res = await fetch(`${API}/buses/vehicle/${vehicle.vehicle_id}/occupancy`)
      if (res.ok) set({ vehicleOccupancy: await res.json() })
    } catch (_) {
      set({ vehicleOccupancy: { occupancy_level: 'unknown', factors: {} } })
    }
  },

  clearSelectedVehicle: () => set({ selectedVehicle: null, vehicleOccupancy: null }),

  // ── Weather / model status ────────────────────────────────────────────────
  weather: null,
  modelStatus: null,

  fetchStatus: async () => {
    try {
      const res = await fetch(`${API}/predictions/status`)
      if (!res.ok) return
      const status = await res.json()
      set({ weather: status.weather, modelStatus: status })
    } catch (_) {}
  },

  // ── Heatmap ───────────────────────────────────────────────────────────────
  heatmapData: null,
  showHeatmap: false,

  toggleHeatmap: async () => {
    const next = !get().showHeatmap
    set({ showHeatmap: next })
    if (next && !get().heatmapData) {
      try {
        const res = await fetch(`${API}/buses/heatmap`)
        if (!res.ok) return
        const data = await res.json()
        const points = (data.features || []).map(f => ({
          lng: f.geometry.coordinates[0],
          lat: f.geometry.coordinates[1],
          students: f.properties.students || 0,
          weight: f.properties.weight || 0,
        }))
        set({ heatmapData: points })
      } catch (_) {}
    }
  },

  // ── Map loaded state ─────────────────────────────────────────────────────────
  mapLoaded: false,
  setMapLoaded: (v) => set({ mapLoaded: v }),

  // ── Route multi-select filter ─────────────────────────────────────────────
  // Stored as plain object { route_id: true } so Zustand detects changes.
  // Empty object = ALL routes visible.
  activeRoutesMap: {},
  showRoutes: true,
  showStops: true,

  toggleRouteFilter: (route_id) => {
    set((state) => {
      const next = { ...state.activeRoutesMap }
      if (next[route_id]) {
        delete next[route_id]
      } else {
        next[route_id] = true
      }
      // Debug: log what's selected
      console.log('[RouteFilter] activeRoutesMap:', next)
      return { activeRoutesMap: next }
    })
  },

  clearRouteFilter: () => set({ activeRoutesMap: {} }),
  toggleRoutes:     () => set((s) => ({ showRoutes: !s.showRoutes })),
  toggleStops:      () => set((s) => ({ showStops:  !s.showStops  })),
}))
