"use client"

import { useExplorerStore } from "@/lib/explorer-store"
import { X } from "lucide-react"
import { Button } from "@/components/ui/button"

export function PlanetInfo() {
  const { selectedPlanet, setSelectedPlanet } = useExplorerStore()

  if (!selectedPlanet) return null

  return (
    <div className="fixed bottom-4 left-1/2 -translate-x-1/2 w-full max-w-2xl px-4 z-50">
      <div className="bg-card border border-border rounded-lg p-6 shadow-2xl animate-in slide-in-from-bottom-4">
        <div className="flex items-start justify-between mb-4">
          <div>
            <h2 className="text-2xl font-bold">{selectedPlanet.name}</h2>
            <p className="text-sm text-muted-foreground">{selectedPlanet.type} Planet</p>
          </div>
          <Button variant="ghost" size="icon" onClick={() => setSelectedPlanet(null)}>
            <X className="w-4 h-4" />
          </Button>
        </div>

        <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
          <div className="space-y-1">
            <p className="text-sm text-muted-foreground">Radius</p>
            <p className="text-lg font-semibold">{selectedPlanet.radius.toFixed(2)} R⊕</p>
          </div>

          <div className="space-y-1">
            <p className="text-sm text-muted-foreground">Distance</p>
            <p className="text-lg font-semibold">{selectedPlanet.distance.toFixed(4)} AU</p>
          </div>

          <div className="space-y-1">
            <p className="text-sm text-muted-foreground">Orbital Period</p>
            <p className="text-lg font-semibold">{selectedPlanet.period.toFixed(2)} days</p>
          </div>

          <div className="space-y-1">
            <p className="text-sm text-muted-foreground">Temperature</p>
            <p className="text-lg font-semibold">{selectedPlanet.temperature}K</p>
          </div>

          <div className="space-y-1">
            <p className="text-sm text-muted-foreground">Discovered</p>
            <p className="text-lg font-semibold">{selectedPlanet.discovered}</p>
          </div>

          <div className="space-y-1">
            <p className="text-sm text-muted-foreground">Type</p>
            <p className="text-lg font-semibold">{selectedPlanet.type}</p>
          </div>
        </div>

        <p className="text-sm text-muted-foreground mt-4">
          Click another planet to view its details, or click empty space to close this panel.
        </p>
      </div>
    </div>
  )
}
