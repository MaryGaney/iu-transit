// src/App.jsx
import { useEffect } from 'react'
import { useTransitStore } from './store/transitStore'
import MapView from './components/MapView'
import TopBar from './components/TopBar'
import StopPanel from './components/StopPanel'
import StatusBar from './components/StatusBar'
import TravelAgent from './components/TravelAgent'
import './styles/global.css'

export default function App() {
  const { fetchRoutes, fetchStops, fetchStatus, connectWebSocket } = useTransitStore()

  useEffect(() => {
    fetchRoutes()
    fetchStops()
    fetchStatus()
    connectWebSocket()
    const interval = setInterval(fetchStatus, 60_000)
    return () => clearInterval(interval)
  }, [])

  return (
    <div className="app-shell">
      <TopBar />
      <div className="map-container">
        {/* MapView renders VehiclePopup internally when a bus is selected */}
        <MapView />
        {/* StopPanel renders when a stop is selected */}
        <StopPanel />
      </div>
      <StatusBar />
      <TravelAgent />
    </div>
  )
}
