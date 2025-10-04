"use client"

import { useState } from "react"
import { Card } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { fetchExoplanetSystem, type SystemData } from "@/lib/nasa-api"
import { Loader2, Database } from "lucide-react"

export function NasaDataLoader() {
  const [systemData, setSystemData] = useState<SystemData | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const loadSystem = async (systemName: string) => {
    setLoading(true)
    setError(null)
    try {
      const data = await fetchExoplanetSystem(systemName)
      setSystemData(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load system data")
    } finally {
      setLoading(false)
    }
  }

  return (
    <Card className="p-6 border-border">
      <div className="flex items-center gap-2 mb-4">
        <Database className="w-5 h-5 text-primary" />
        <h3 className="text-xl font-bold">NASA Exoplanet Data</h3>
      </div>

      <div className="space-y-4">
        <div className="flex gap-2">
          <Button onClick={() => loadSystem("TRAPPIST-1")} disabled={loading} size="sm">
            Load TRAPPIST-1
          </Button>
          <Button onClick={() => loadSystem("Kepler-90")} disabled={loading} size="sm" variant="outline">
            Load Kepler-90
          </Button>
          <Button onClick={() => loadSystem("TOI-700")} disabled={loading} size="sm" variant="outline">
            Load TOI-700
          </Button>
        </div>

        {loading && (
          <div className="flex items-center gap-2 text-muted-foreground">
            <Loader2 className="w-4 h-4 animate-spin" />
            <span className="text-sm">Loading data from NASA Exoplanet Archive...</span>
          </div>
        )}

        {error && (
          <div className="p-4 bg-destructive/10 border border-destructive/20 rounded-lg">
            <p className="text-sm text-destructive">{error}</p>
          </div>
        )}

        {systemData && (
          <div className="space-y-3">
            <div className="p-4 bg-secondary/50 rounded-lg">
              <h4 className="font-semibold mb-2">{systemData.system}</h4>
              <p className="text-sm text-muted-foreground">
                {systemData.count} planet{systemData.count !== 1 ? "s" : ""} confirmed
              </p>
            </div>

            <div className="space-y-2 max-h-64 overflow-y-auto">
              {systemData.planets.map((planet, index) => (
                <div key={index} className="p-3 bg-card border border-border rounded-lg text-sm">
                  <div className="font-semibold">{planet.name}</div>
                  <div className="grid grid-cols-2 gap-2 mt-2 text-xs text-muted-foreground">
                    <div>Radius: {planet.radius.toFixed(2)} R⊕</div>
                    <div>Period: {planet.period.toFixed(2)} days</div>
                    <div>Temp: {planet.temperature.toFixed(0)}K</div>
                    <div>Discovered: {planet.discovered}</div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </Card>
  )
}
