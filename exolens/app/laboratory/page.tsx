'use client'

import { useState } from "react"
import { Card } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Telescope, FlaskConical, Earth, ChevronRight } from "lucide-react"
import { ExoplanetExplorer } from "@/components/exoplanet-explorer"
import { PlanetCreator } from "@/components/planet-creator"
import { EarthSimilarityCalculator } from "@/components/earth-similarity-calculator"

type LabMode = 'explorer' | 'detector' | 'similarity'

export interface PlanetData {
  name: string
  radius: number
  mass: number
  temperature: number
  escape_velocity: number
  density?: number
  orbital_period?: number
  distance_from_star?: number
}

export default function LaboratoryPage() {
  const [activeMode, setActiveMode] = useState<LabMode>('explorer')
  const [selectedPlanet, setSelectedPlanet] = useState<PlanetData | null>(null)

  const handlePlanetSelect = (planetData: PlanetData) => {
    setSelectedPlanet(planetData)
    setActiveMode('similarity')
  }

  const modes = [
    {
      id: 'explorer' as LabMode,
      title: "Exoplanet Explorer",
      description: "Browse stars and their confirmed exoplanets from NASA database",
      icon: Telescope,
      color: "from-purple-500 to-blue-500",
      hoverColor: "hover:from-purple-600 hover:to-blue-600"
    },
    {
      id: 'detector' as LabMode,
      title: "Exoplanet Detector",
      description: "AI-powered detection using features or light curve analysis",
      icon: FlaskConical,
      color: "from-blue-500 to-cyan-500",
      hoverColor: "hover:from-blue-600 hover:to-cyan-600"
    },
    {
      id: 'similarity' as LabMode,
      title: "Earth Similarity Index",
      description: "Calculate how Earth-like an exoplanet is based on physical properties",
      icon: Earth,
      color: "from-green-500 to-emerald-500",
      hoverColor: "hover:from-green-600 hover:to-emerald-600"
    }
  ]

  return (
    <div className="min-h-screen pt-16 pb-12 bg-gradient-to-b from-black via-[#0a0a0a] to-black">
      <div className="container mx-auto px-4 py-8">
        <div className="max-w-7xl mx-auto space-y-8">

          {/* Header */}
          <div className="text-center space-y-4">
            <h1 className="text-4xl md:text-5xl font-bold bg-gradient-to-r from-blue-400 via-purple-400 to-pink-400 bg-clip-text text-transparent">
              Planetary Laboratory
            </h1>
            <p className="text-lg text-gray-400 max-w-2xl mx-auto">
              Explore, detect, and analyze exoplanets using cutting-edge AI and NASA data
            </p>
          </div>

          {/* Mode Selection Buttons */}
          <div className="grid md:grid-cols-3 gap-4">
            {modes.map((mode) => {
              const Icon = mode.icon
              const isActive = activeMode === mode.id

              return (
                <button
                  key={mode.id}
                  onClick={() => setActiveMode(mode.id)}
                  className={`
                    relative group p-6 rounded-xl transition-all duration-300
                    ${isActive
                      ? `bg-gradient-to-br ${mode.color} shadow-2xl scale-105`
                      : 'bg-[#1b1b1b] hover:bg-[#252525] border border-[#333333]'
                    }
                  `}
                >
                  <div className="flex flex-col items-center text-center space-y-3">
                    <div className={`
                      p-4 rounded-full transition-all duration-300
                      ${isActive
                        ? 'bg-white/20'
                        : 'bg-[#2c2c2c] group-hover:bg-[#3c3c3c]'
                      }
                    `}>
                      <Icon className={`w-8 h-8 ${isActive ? 'text-white' : 'text-gray-400'}`} />
                    </div>

                    <div>
                      <h3 className={`text-lg font-bold mb-1 ${isActive ? 'text-white' : 'text-gray-300'}`}>
                        {mode.title}
                      </h3>
                      <p className={`text-sm ${isActive ? 'text-white/80' : 'text-gray-500'}`}>
                        {mode.description}
                      </p>
                    </div>

                    {isActive && (
                      <div className="absolute bottom-3 right-3">
                        <ChevronRight className="w-5 h-5 text-white animate-pulse" />
                      </div>
                    )}
                  </div>
                </button>
              )
            })}
          </div>

          {/* Active Component */}
          <Card className="p-6 bg-[#1b1b1b] border-[#333333] shadow-2xl">
            {activeMode === 'explorer' && <ExoplanetExplorer onPlanetSelect={handlePlanetSelect} />}
            {activeMode === 'detector' && <PlanetCreator />}
            {activeMode === 'similarity' && (
              <EarthSimilarityCalculator
                initialData={selectedPlanet}
                onClear={() => setSelectedPlanet(null)}
              />
            )}
          </Card>
        </div>
      </div>
    </div>
  )
}
