import { create } from "zustand"

interface PlanetData {
  name: string
  type: string
  radius: number
  distance: number
  period: number
  temperature: number
  discovered: string
}

interface ExplorerStore {
  selectedPlanet: PlanetData | null
  setSelectedPlanet: (planet: PlanetData | null) => void
}

export const useExplorerStore = create<ExplorerStore>((set) => ({
  selectedPlanet: null,
  setSelectedPlanet: (planet) => set({ selectedPlanet: planet }),
}))
