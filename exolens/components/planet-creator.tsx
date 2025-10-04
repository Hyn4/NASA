"use client"

import { useState } from "react"
import { Card } from "@/components/ui/card"
import { Label } from "@/components/ui/label"
import { Slider } from "@/components/ui/slider"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Button } from "@/components/ui/button"
import { PlanetPreview } from "@/components/planet-preview"
import { LightCurveSimulation } from "@/components/light-curve-simulation"
import { Sparkles } from "lucide-react"
import { generatePlanetTexture } from "@/lib/nasa-api"
import { useToast } from "@/hooks/use-toast"

export interface PlanetParameters {
  radius: number // Earth radii
  mass: number // Earth masses
  temperature: number // Kelvin
  starType: "red-dwarf" | "sun-like" | "blue-giant"
  distance: number // AU
}

const starTypes = {
  "red-dwarf": { name: "Red Dwarf", temp: 3000, color: "#ff6b4a" },
  "sun-like": { name: "Sun-like Star", temp: 5800, color: "#ffeb3b" },
  "blue-giant": { name: "Blue Giant", temp: 10000, color: "#64b5f6" },
}

export function PlanetCreator() {
  const [parameters, setParameters] = useState<PlanetParameters>({
    radius: 1.0,
    mass: 1.0,
    temperature: 288,
    starType: "sun-like",
    distance: 1.0,
  })

  const [isGenerating, setIsGenerating] = useState(false)
  const [generatedTexture, setGeneratedTexture] = useState<{ url: string; prompt: string } | null>(null)
  const { toast } = useToast()

  const updateParameter = <K extends keyof PlanetParameters>(key: K, value: PlanetParameters[K]) => {
    setParameters((prev) => ({ ...prev, [key]: value }))
  }

  const handleGenerate = async () => {
    setIsGenerating(true)
    try {
      const result = await generatePlanetTexture({
        radius: parameters.radius,
        temperature: parameters.temperature,
        mass: parameters.mass,
        starType: parameters.starType,
        planetType: planetType,
      })

      setGeneratedTexture(result)

      toast({
        title: "Planet Generated!",
        description: "Your custom planet has been created with AI-generated textures.",
      })
    } catch (error) {
      toast({
        title: "Generation Failed",
        description: "Failed to generate planet texture. Please try again.",
        variant: "destructive",
      })
    } finally {
      setIsGenerating(false)
    }
  }

  const planetType =
    parameters.radius < 1.5
      ? "Terrestrial"
      : parameters.radius < 2.5
        ? "Super-Earth"
        : parameters.radius < 6
          ? "Neptune-like"
          : "Gas Giant"

  const habitableZone =
    parameters.distance > 0.5 &&
    parameters.distance < 2.0 &&
    parameters.temperature > 273 &&
    parameters.temperature < 373

  return (
    <div className="max-w-7xl mx-auto">
      <div className="grid lg:grid-cols-2 gap-8">
        {/* Controls Panel */}
        <Card className="p-6 space-y-6 border-border">
          <div className="space-y-2">
            <h2 className="text-2xl font-bold">Planet Parameters</h2>
            <p className="text-sm text-muted-foreground">Adjust the sliders to design your custom exoplanet</p>
          </div>

          {/* Radius */}
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <Label htmlFor="radius">Planet Radius</Label>
              <span className="text-sm font-mono text-muted-foreground">{parameters.radius.toFixed(2)} R⊕</span>
            </div>
            <Slider
              id="radius"
              min={0.5}
              max={15}
              step={0.1}
              value={[parameters.radius]}
              onValueChange={([value]) => updateParameter("radius", value)}
              className="w-full"
            />
            <p className="text-xs text-muted-foreground">Earth radii (R⊕)</p>
          </div>

          {/* Mass */}
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <Label htmlFor="mass">Planet Mass</Label>
              <span className="text-sm font-mono text-muted-foreground">{parameters.mass.toFixed(2)} M⊕</span>
            </div>
            <Slider
              id="mass"
              min={0.1}
              max={100}
              step={0.5}
              value={[parameters.mass]}
              onValueChange={([value]) => updateParameter("mass", value)}
              className="w-full"
            />
            <p className="text-xs text-muted-foreground">Earth masses (M⊕)</p>
          </div>

          {/* Temperature */}
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <Label htmlFor="temperature">Surface Temperature</Label>
              <span className="text-sm font-mono text-muted-foreground">{parameters.temperature}K</span>
            </div>
            <Slider
              id="temperature"
              min={100}
              max={1500}
              step={10}
              value={[parameters.temperature]}
              onValueChange={([value]) => updateParameter("temperature", value)}
              className="w-full"
            />
            <p className="text-xs text-muted-foreground">Kelvin (K) - Water freezes at 273K, boils at 373K</p>
          </div>

          {/* Star Type */}
          <div className="space-y-3">
            <Label htmlFor="starType">Host Star Type</Label>
            <Select value={parameters.starType} onValueChange={(value: any) => updateParameter("starType", value)}>
              <SelectTrigger id="starType">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {Object.entries(starTypes).map(([key, value]) => (
                  <SelectItem key={key} value={key}>
                    {value.name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          {/* Distance */}
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <Label htmlFor="distance">Distance from Star</Label>
              <span className="text-sm font-mono text-muted-foreground">{parameters.distance.toFixed(2)} AU</span>
            </div>
            <Slider
              id="distance"
              min={0.1}
              max={5}
              step={0.1}
              value={[parameters.distance]}
              onValueChange={([value]) => updateParameter("distance", value)}
              className="w-full"
            />
            <p className="text-xs text-muted-foreground">Astronomical Units (AU) - Earth is 1 AU from the Sun</p>
          </div>

          {/* Planet Classification */}
          <Card className="p-4 bg-secondary/50 border-border">
            <div className="space-y-2">
              <h3 className="font-semibold text-sm">Planet Classification</h3>
              <div className="flex items-center justify-between">
                <span className="text-sm text-muted-foreground">Type:</span>
                <span className="text-sm font-bold text-primary">{planetType}</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm text-muted-foreground">Habitable Zone:</span>
                <span className={`text-sm font-bold ${habitableZone ? "text-green-500" : "text-muted-foreground"}`}>
                  {habitableZone ? "Yes" : "No"}
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm text-muted-foreground">Density:</span>
                <span className="text-sm font-mono">
                  {(parameters.mass / Math.pow(parameters.radius, 3)).toFixed(2)} g/cm³
                </span>
              </div>
            </div>
          </Card>

          {/* Generate Button */}
          <Button onClick={handleGenerate} disabled={isGenerating} className="w-full" size="lg">
            {isGenerating ? (
              <>
                <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin mr-2" />
                Generating Planet...
              </>
            ) : (
              <>
                <Sparkles className="w-4 h-4 mr-2" />
                Generate My Planet
              </>
            )}
          </Button>

          {/* AI Generation Prompt */}
          {generatedTexture && (
            <Card className="p-4 bg-accent/10 border-accent/20">
              <h4 className="text-sm font-semibold mb-2">AI Generation Prompt:</h4>
              <p className="text-xs text-muted-foreground leading-relaxed">{generatedTexture.prompt}</p>
            </Card>
          )}
        </Card>

        {/* Preview Panel */}
        <div className="space-y-6">
          <Card className="p-6 border-border">
            <h3 className="text-xl font-bold mb-4">3D Preview</h3>
            <PlanetPreview parameters={parameters} isGenerating={isGenerating} textureUrl={generatedTexture?.url} />
          </Card>

          <Card className="p-6 border-border">
            <h3 className="text-xl font-bold mb-4">Transit Light Curve</h3>
            <p className="text-sm text-muted-foreground mb-4">
              This shows how the star's brightness would change if your planet passed in front of it
            </p>
            <LightCurveSimulation parameters={parameters} />
          </Card>
        </div>
      </div>
    </div>
  )
}
