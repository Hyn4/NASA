import { PlanetCreator } from "@/components/planet-creator"
import { Flag as Flask } from "lucide-react"

export default function LaboratoryPage() {
  return (
    <div className="min-h-screen pt-16 bg-[#1b1b1b]">

      <div className="container mx-auto px-4 py-12">
        {/* Header */}
        <div className="max-w-4xl mx-auto text-center space-y-6 mb-12">
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-accent/10 border border-accent/20 text-sm">
            <Flask className="w-4 h-4 text-white" />
            <span>Interactive Simulator</span>
          </div>

          <h1 className="text-4xl md:text-6xl font-bold text-white">
            Planetary <span className="text-cosmic">Laboratory</span>
          </h1>

          <p className="text-xl text-muted-foreground leading-relaxed">
            Design your own exoplanet by adjusting physical parameters. Watch it come to life in 3D and see what its
            transit signal would look like.
          </p>
        </div>

        {/* Planet Creator */}
        <PlanetCreator />
      </div>
    </div>
  )
}
