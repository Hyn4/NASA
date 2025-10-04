import { ThreeScene } from "@/components/three-scene"
import { PlanetInfo } from "@/components/planet-info"

export default function ExplorerPage() {
  return (
    <div className="min-h-screen pt-16 bg-background">
      <ThreeScene />
      <PlanetInfo />
    </div>
  )
}
