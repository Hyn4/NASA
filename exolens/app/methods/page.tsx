import { PlanetTypes } from "@/components/planet-types"
import { DetectionMethods } from "@/components/detection-methods"
import { Telescope } from "lucide-react"

export default function MethodsPage() {
  return (
    <div className="min-h-screen pt-16 bg-[#1b1b1b]">
      <div className="container mx-auto px-4 py-12">
        {/* Header */}
        <div className="max-w-4xl mx-auto text-center space-y-6 mb-16 mt-4">
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-primary/10 border border-white/20 text-sm">
            <Telescope className="w-4 h-4 text-white" />
            <span className="text-white font-semibold">Educational Content</span>
          </div>

          <h1 className="text-4xl md:text-6xl font-bold text-balance">
            How We <span className="text-cosmic">Find</span> Planets
          </h1>

          <p className="text-xl text-gray-300 leading-relaxed">
            Discovering worlds beyond our solar system requires ingenious techniques. Learn about the different types of
            exoplanets and the methods astronomers use to detect them.
          </p>
        </div>

        {/* Planet Types Section */}
        <PlanetTypes />

        {/* Detection Methods Section */}
        <DetectionMethods />
      </div>
    </div>
  )
}
