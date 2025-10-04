"use client"

import { OrbitControls } from "@react-three/drei"
import { Canvas, useFrame } from "@react-three/fiber"
import { useRef } from "react"
import type * as THREE from "three"
import type { PlanetSimulationParameters } from "./planet-creator"

interface PlanetPreviewProps {
  parameters: PlanetSimulationParameters
  isGenerating: boolean
  textureUrl?: string
}

function Planet({ parameters, textureUrl }: { parameters: PlanetSimulationParameters; textureUrl?: string }) {
  const meshRef = useRef<THREE.Mesh>(null)

  useFrame(() => {
    if (meshRef.current) {
      meshRef.current.rotation.y += 0.005
    }
  })

  // Determine color based on temperature
  const getColor = () => {
    if (parameters.temperature < 200) return "#87ceeb" // Ice blue
    if (parameters.temperature < 273) return "#708090" // Cold gray
    if (parameters.temperature < 373) return "#4682b4" // Earth-like blue
    if (parameters.temperature < 600) return "#d2691e" // Hot brown
    if (parameters.temperature < 1000) return "#ff6347" // Very hot red
    return "#ff4500" // Extreme heat
  }

  // Determine material properties based on planet type
  const getMaterialProps = () => {
    if (parameters.radius > 6) {
      return { metalness: 0.1, roughness: 0.9 } // Gas giant
    } else if (parameters.radius > 2.5) {
      return { metalness: 0.2, roughness: 0.8 } // Ice giant
    }
    return { metalness: 0.3, roughness: 0.7 } // Rocky
  }

  const materialProps = getMaterialProps()

  return (
    <mesh ref={meshRef} scale={parameters.radius * 0.8}>
      <sphereGeometry args={[1, 64, 64]} />
      <meshStandardMaterial color={getColor()} {...materialProps} />
    </mesh>
  )
}

export function PlanetPreview({ parameters, isGenerating, textureUrl }: PlanetPreviewProps) {
  return (
    <div className="relative">
      <div className="w-full  h-[400px] rounded-lg overflow-hidden bg-gradient-to-b to-[#0f0f1a] via-[#0f0f1a] from-[#0f0f1a]">
        
        <Canvas camera={{ position: [0, 0, 5], fov: 75 }}>
          <ambientLight intensity={0.5} />
          <pointLight position={[5, 5, 5]} intensity={1} />
          <Planet parameters={parameters} textureUrl={textureUrl} />
          <OrbitControls enableDamping dampingFactor={0.05} autoRotate autoRotateSpeed={2} />
        </Canvas>
      </div>
      {isGenerating && (
        <div className="absolute inset-0 flex items-center justify-center bg-background/80 backdrop-blur-sm rounded-lg">
          <div className="text-center space-y-4">
            <div className="w-12 h-12 border-4 border-primary border-t-transparent rounded-full animate-spin mx-auto" />
            <p className="text-sm text-muted-foreground">Generating planet texture...</p>
          </div>
        </div>
      )}
    </div>
  )
}
