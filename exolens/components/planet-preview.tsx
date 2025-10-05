'use client'

import { useRef, useEffect, useState } from "react"
import { Canvas, useFrame } from "@react-three/fiber"
import { OrbitControls } from "@react-three/drei"
import * as THREE from "three"
import type { PlanetSimulationParameters } from "./planet-creator"
import { StarField3D } from "./star-field-3d"

interface PlanetPreviewProps {
  parameters: PlanetSimulationParameters
  isGenerating: boolean
  textureUrl?: string
}

function Planet({ parameters, textureUrl }: {
  parameters: PlanetSimulationParameters
  textureUrl?: string
}) {
  const meshRef = useRef<THREE.Mesh>(null!)
  const materialRef = useRef<THREE.MeshStandardMaterial>(null!)
  const [texture, setTexture] = useState<THREE.Texture | null>(null)

  useEffect(() => {
    if (!textureUrl || !textureUrl.startsWith('data:image')) {
      setTexture(null);
      return;
    }

    const loader = new THREE.TextureLoader()
    loader.load(
      textureUrl,
      (loadedTexture) => {
        // --- INÍCIO DA CORREÇÃO ---
        // CORREÇÃO 1: 'encoding' foi substituído por 'colorSpace' nas versões recentes do Three.js
        // CORREÇÃO 2: O valor agora é 'SRGBColorSpace'
        loadedTexture.colorSpace = THREE.SRGBColorSpace; 
        // --- FIM DA CORREÇÃO ---

        loadedTexture.flipY = false;
        setTexture(loadedTexture)
      },
      undefined,
      (error) => {
        console.error('Error loading texture:', error)
      }
    )

    return () => {
      texture?.dispose();
    }
  }, [textureUrl])

  useFrame(() => {
    if (meshRef.current) {
      meshRef.current.rotation.y += 0.005
    }
  })
  
  // As suas funções de cor e material continuam aqui, sem alterações...
  const lerp = (start: number, end: number, t: number) => {
    return start + (end - start) * t
  }

  const lerpColor = (color1: string, color2: string, t: number) => {
    const c1 = new THREE.Color(color1)
    const c2 = new THREE.Color(color2)
    return new THREE.Color(
      lerp(c1.r, c2.r, t),
      lerp(c1.g, c2.g, t),
      lerp(c1.b, c2.b, t)
    )
  }

  const getColor = () => {
    const temp = parameters.temperature

    const colorStops = [
      { temp: 0, color: "#0a2877" },
      { temp: 100, color: "#1e3a5f" },
      { temp: 200, color: "#87ceeb" },
      { temp: 273, color: "#4682b4" },
      { temp: 373, color: "#5f9ea0" },
      { temp: 500, color: "#8b7355" },
      { temp: 600, color: "#d2691e" },
      { temp: 800, color: "#ff8c00" },
      { temp: 1000, color: "#ff6347" },
      { temp: 1200, color: "#ff4500" },
      { temp: 1500, color: "#ff0000" },
      { temp: 2000, color: "#ffff00" },
      { temp: 3000, color: "#ffffff" }
    ]

    let lower = colorStops[0]
    let upper = colorStops[colorStops.length - 1]

    for (let i = 0; i < colorStops.length - 1; i++) {
      if (temp >= colorStops[i].temp && temp <= colorStops[i + 1].temp) {
        lower = colorStops[i]
        upper = colorStops[i + 1]
        break
      }
    }

    if (temp <= colorStops[0].temp) return colorStops[0].color
    if (temp >= colorStops[colorStops.length - 1].temp) return colorStops[colorStops.length - 1].color

    const t = (temp - lower.temp) / (upper.temp - lower.temp)
    const interpolatedColor = lerpColor(lower.color, upper.color, t)
    return `#${interpolatedColor.getHexString()}`
  }

  const getMaterialProps = () => {
    if (parameters.radius > 6) {
      return { metalness: 0.1, roughness: 0.9 }
    } else if (parameters.radius > 2.5) {
      return { metalness: 0.2, roughness: 0.8 }
    }
    return { metalness: 0.3, roughness: 0.7 }
  }


  const materialProps = getMaterialProps()

  useEffect(() => {
    if (materialRef.current) {
      if (texture) {
        materialRef.current.map = texture;
        materialRef.current.color.set("white");
      } else {
        materialRef.current.map = null;
        materialRef.current.color.set(getColor());
      }
      materialRef.current.needsUpdate = true;
    }
  }, [texture, parameters.temperature]);


  return (
    <mesh ref={meshRef} scale={parameters.radius * 0.8}>
      <sphereGeometry args={[1, 64, 64]} />
      <meshStandardMaterial ref={materialRef} {...materialProps} />
    </mesh>
  )
}

// O componente PlanetPreview continua igual.
export function PlanetPreview({ parameters, isGenerating, textureUrl }: PlanetPreviewProps) {
  const [mounted, setMounted] = useState(false)

  useEffect(() => {
    setMounted(true)
  }, [])

  if (!mounted) {
    return (
      <div className="w-full h-[400px] rounded-lg overflow-hidden bg-black flex items-center justify-center">
        <p className="text-gray-400">Loading 3D Preview...</p>
      </div>
    )
  }

  return (
    <div className="relative">
      <div className="w-full h-[400px] rounded-lg overflow-hidden bg-black">
        <Canvas camera={{ position: [0, 0, 5], fov: 75 }}>
          <StarField3D />
          <ambientLight intensity={0.5} />
          <pointLight position={[5, 5, 5]} intensity={1} />
          <Planet parameters={parameters} textureUrl={textureUrl} />
          <OrbitControls
            enableDamping
            dampingFactor={0.05}
            autoRotate
            autoRotateSpeed={2}
          />
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