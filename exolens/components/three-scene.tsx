"use client"

import { useRef, useState } from "react"
import { Canvas, useFrame } from "@react-three/fiber"
import { OrbitControls, Stars } from "@react-three/drei"
import * as THREE from "three"
import { useExplorerStore } from "@/lib/explorer-store"

// TRAPPIST-1 system data (simplified for MVP)
const TRAPPIST1_DATA = {
  star: {
    name: "TRAPPIST-1",
    radius: 0.117,
    color: "#ff6b4a",
  },
  planets: [
    { name: "TRAPPIST-1 b", radius: 1.116, distance: 0.01154, period: 1.51, color: "#8b7355", type: "Terrestrial" },
    { name: "TRAPPIST-1 c", radius: 1.097, distance: 0.0158, period: 2.42, color: "#a0826d", type: "Terrestrial" },
    { name: "TRAPPIST-1 d", radius: 0.788, distance: 0.02227, period: 4.05, color: "#6b8e23", type: "Terrestrial" },
    { name: "TRAPPIST-1 e", radius: 0.92, distance: 0.02925, period: 6.1, color: "#4682b4", type: "Terrestrial" },
    { name: "TRAPPIST-1 f", radius: 1.045, distance: 0.03849, period: 9.21, color: "#5f9ea0", type: "Terrestrial" },
    { name: "TRAPPIST-1 g", radius: 1.129, distance: 0.04683, period: 12.35, color: "#708090", type: "Terrestrial" },
    { name: "TRAPPIST-1 h", radius: 0.755, distance: 0.06189, period: 18.77, color: "#778899", type: "Terrestrial" },
  ],
}

function Star() {
  const meshRef = useRef<THREE.Mesh>(null)

  useFrame(() => {
    if (meshRef.current) {
      meshRef.current.rotation.y += 0.001
    }
  })

  return (
    <group>
      <mesh ref={meshRef}>
        <sphereGeometry args={[2, 32, 32]} />
        <meshBasicMaterial color={TRAPPIST1_DATA.star.color} />
      </mesh>
      <mesh>
        <sphereGeometry args={[2.3, 32, 32]} />
        <meshBasicMaterial color={TRAPPIST1_DATA.star.color} transparent opacity={0.3} />
      </mesh>
      <pointLight color={TRAPPIST1_DATA.star.color} intensity={2} distance={100} />
    </group>
  )
}

function Orbit({ distance }: { distance: number }) {
  const points = []
  for (let i = 0; i <= 64; i++) {
    const angle = (i / 64) * Math.PI * 2
    points.push(new THREE.Vector3(Math.cos(angle) * distance, 0, Math.sin(angle) * distance))
  }

  const lineGeometry = new THREE.BufferGeometry().setFromPoints(points)

  return (
    <line geometry={lineGeometry}>
      <lineBasicMaterial color="#444444" transparent opacity={0.3} />
    </line>
  )
}

function Planet({
  data,
  index,
  onClick,
}: {
  data: (typeof TRAPPIST1_DATA.planets)[0]
  index: number
  onClick: () => void
}) {
  const meshRef = useRef<THREE.Mesh>(null)
  const distance = data.distance * 100
  const angleRef = useRef((index / TRAPPIST1_DATA.planets.length) * Math.PI * 2)

  useFrame(() => {
    if (meshRef.current) {
      angleRef.current += (0.001 / data.period) * 10
      meshRef.current.position.x = Math.cos(angleRef.current) * distance
      meshRef.current.position.z = Math.sin(angleRef.current) * distance
    }
  })

  return (
    <>
      <Orbit distance={distance} />
      <mesh ref={meshRef} onClick={onClick}>
        <sphereGeometry args={[data.radius * 0.5, 32, 32]} />
        <meshStandardMaterial color={data.color} roughness={0.7} metalness={0.3} />
      </mesh>
    </>
  )
}

function Scene() {
  const { setSelectedPlanet } = useExplorerStore()

  const handlePlanetClick = (planetData: (typeof TRAPPIST1_DATA.planets)[0], index: number) => {
    setSelectedPlanet({
      name: planetData.name,
      type: planetData.type,
      radius: planetData.radius,
      distance: planetData.distance,
      period: planetData.period,
      temperature: 200 + index * 20,
      discovered: "2017",
    })
  }

  return (
    <>
      <color attach="background" args={["#0a0a0f"]} />
      <ambientLight intensity={0.3} />
      <Stars radius={100} depth={50} count={1000} factor={4} saturation={0} fade speed={1} />
      <Star />
      {TRAPPIST1_DATA.planets.map((planet, index) => (
        <Planet key={planet.name} data={planet} index={index} onClick={() => handlePlanetClick(planet, index)} />
      ))}
      <OrbitControls enableDamping dampingFactor={0.05} minDistance={5} maxDistance={100} makeDefault />
    </>
  )
}

export function ThreeScene() {
  const [isLoading, setIsLoading] = useState(true)

  return (
    <div className="w-full h-screen relative">
      {isLoading && (
        <div className="absolute inset-0 flex items-center justify-center bg-background z-10">
          <div className="text-center space-y-4">
            <div className="w-16 h-16 border-4 border-primary border-t-transparent rounded-full animate-spin mx-auto" />
            <p className="text-muted-foreground">Loading TRAPPIST-1 System...</p>
          </div>
        </div>
      )}

      <Canvas camera={{ position: [0, 15, 25], fov: 75 }} onCreated={() => setIsLoading(false)}>
        <Scene />
      </Canvas>

      {/* Instructions overlay */}
      <div className="absolute top-4 left-4 bg-card/80 backdrop-blur-sm border border-border rounded-lg p-4 max-w-xs">
        <h3 className="font-bold mb-2">Controls</h3>
        <ul className="text-sm text-muted-foreground space-y-1">
          <li>• Click and drag to rotate</li>
          <li>• Scroll to zoom</li>
          <li>• Click a planet for details</li>
        </ul>
      </div>

      {/* System info */}
      <div className="absolute top-4 right-4 bg-card/80 backdrop-blur-sm border border-border rounded-lg p-4">
        <h3 className="font-bold text-lg mb-2">TRAPPIST-1 System</h3>
        <div className="text-sm space-y-1">
          <p className="text-muted-foreground">
            <span className="text-foreground font-medium">Distance:</span> 40 light-years
          </p>
          <p className="text-muted-foreground">
            <span className="text-foreground font-medium">Planets:</span> 7 terrestrial
          </p>
          <p className="text-muted-foreground">
            <span className="text-foreground font-medium">Star Type:</span> Ultra-cool red dwarf
          </p>
        </div>
      </div>
    </div>
  )
}
