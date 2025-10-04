"use client"

import { useState, useEffect } from "react"
import { Card } from "@/components/ui/card"
import { Label } from "@/components/ui/label"
import { Input } from "@/components/ui/input"
import { Button } from "@/components/ui/button"
import { PlanetPreview } from "@/components/planet-preview"
import { LightCurveSimulation } from "@/components/light-curve-simulation"
import { Sparkles, Info } from "lucide-react"
import { generatePlanetTexture } from "@/lib/nasa-api"
import { useToast } from "@/hooks/use-toast"
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip"

// Interface for the input fields
export interface PlanetAnalysisParameters {
  pl_orbper: string // Orbital Period (days)
  pl_rade: string // Planet Radius (Earth Radius)
  pl_trandep: string // Transit Depth (ppm)
  st_teff: string // Stellar Effective Temperature (K)
  st_rad: string // Stellar Radius (Solar Radius)
  st_logg: string // Stellar Surface Gravity (log10(cm/s**2))
}

// Interface for the simulation components (PlanetPreview, LightCurve)
export interface PlanetSimulationParameters {
  radius: number
  mass: number
  temperature: number
  starType: "red-dwarf" | "sun-like" | "blue-giant"
  distance: number
}

const parameterLabels: { [key in keyof PlanetAnalysisParameters]: { label: string; tooltip: string } } = {
  pl_orbper: { label: "Orbital Period (days)", tooltip: "Time the planet takes to orbit its star." },
  pl_rade: { label: "Planet Radius (R⊕)", tooltip: "The radius of the planet in multiples of Earth's radius." },
  pl_trandep: { label: "Transit Depth (ppm)", tooltip: "The percentage of the star's light blocked by the planet, in parts per million." },
  st_teff: { label: "Stellar Temperature (K)", tooltip: "The effective surface temperature of the host star in Kelvin." },
  st_rad: { label: "Stellar Radius (R☉)", tooltip: "The radius of the host star in multiples of the Sun's radius." },
  st_logg: { label: "Stellar Gravity (log(g))", tooltip: "The stellar surface gravity as a logarithm (base 10)." },
}

export function PlanetCreator() {
  // State for the user inputs (with Earth/Sun as default)
  const [analysisParams, setAnalysisParams] = useState<PlanetAnalysisParameters>({
    pl_orbper: "365.25",
    pl_rade: "1",
    pl_trandep: "84",
    st_teff: "5778",
    st_rad: "1",
    st_logg: "4.44",
  })

  // State for the derived simulation parameters
  const [simulationParams, setSimulationParams] = useState<PlanetSimulationParameters>({
    radius: 1,
    mass: 1,
    temperature: 5778,
    starType: "sun-like",
    distance: 1,
  })

  const [isProcessing, setIsProcessing] = useState(false)
  const [generatedTexture, setGeneratedTexture] = useState<{ url: string; prompt: string } | null>(null)
  const [analysisResult, setAnalysisResult] = useState<any>(null)
  const { toast } = useToast()

  // This effect hook derives simulation params from analysis params
  useEffect(() => {
    const deriveSimulationParameters = () => {
      const pl_rade = parseFloat(analysisParams.pl_rade) || 0
      const st_teff = parseFloat(analysisParams.st_teff) || 0
      const st_rad = parseFloat(analysisParams.st_rad) || 0
      const st_logg = parseFloat(analysisParams.st_logg) || 0
      const pl_orbper = parseFloat(analysisParams.pl_orbper) || 0

      if (!pl_rade || !st_teff || !st_rad || !st_logg || !pl_orbper) {
        return // Not enough data to derive
      }

      // 1. Derive Star Type from Temperature
      const starType: PlanetSimulationParameters["starType"] =
        st_teff < 4000 ? "red-dwarf" : st_teff < 7000 ? "sun-like" : "blue-giant"

      // 2. Derive Mass (very rough approximation from radius)
      const mass = Math.pow(pl_rade, 2) // M proportional to R^2 is a loose fit

      // 3. Derive Distance from Orbital Period (Kepler's 3rd Law)
      const g_star_cgs = Math.pow(10, st_logg)
      const r_star_cm = st_rad * 6.957e10
      const G_cgs = 6.6743e-8
      const m_star_g = (g_star_cgs * Math.pow(r_star_cm, 2)) / G_cgs
      const m_star_solar = m_star_g / 1.989e33
      
      const p_years = pl_orbper / 365.25
      const distance_au = Math.cbrt(Math.pow(p_years, 2) * m_star_solar)

      setSimulationParams({
        radius: pl_rade,
        mass: isNaN(mass) ? 1 : mass,
        temperature: st_teff, // Use stellar temp for texture generation
        starType: starType,
        distance: isNaN(distance_au) ? 1 : distance_au,
      })
    }

    deriveSimulationParameters()
  }, [analysisParams])

  const updateParameter = (key: keyof PlanetAnalysisParameters, value: string) => {
    if (/^[0-9]*\.?[0-9]*$/.test(value)) {
      setAnalysisParams((prev) => ({ ...prev, [key]: value }))
    }
  }

  const handleAnalyze = async () => {
    setIsProcessing(true)
    setAnalysisResult(null)
    setGeneratedTexture(null)

    const payload = Object.fromEntries(
      Object.entries(analysisParams).map(([key, value]) => [key, Number(value)])
    )

    if (Object.values(payload).some(v => isNaN(v) || v === 0)) {
        toast({ title: "Invalid Input", description: "Please ensure all fields are filled with valid numbers.", variant: "destructive" })
        setIsProcessing(false)
        return
    }

    try {
      const planetType = simulationParams.radius < 1.5 ? "Terrestrial" : simulationParams.radius < 6 ? "Neptune-like" : "Gas Giant"

      const [textureResult, analysisResponse] = await Promise.all([
        generatePlanetTexture({
          radius: simulationParams.radius,
          temperature: simulationParams.temperature,
          mass: simulationParams.mass,
          starType: simulationParams.starType,
          planetType: planetType,
        }),
        fetch("/api/analyze-planet", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        }),
      ])

      setGeneratedTexture(textureResult)
      toast({ title: "Planet Texture Generated!", description: "AI is creating a visual representation." })

      if (!analysisResponse.ok) {
        const errorData = await analysisResponse.json()
        throw new Error(errorData.error || "Analysis request failed")
      }
      const analysisData = await analysisResponse.json()
      setAnalysisResult(analysisData)
      toast({ title: "Analysis Complete!", description: "The AI has classified the exoplanet candidate." })

    } catch (error: any) {
      toast({ title: "An Error Occurred", description: error.message, variant: "destructive" })
    } finally {
      setIsProcessing(false)
    }
  }

  return (
    <div className="max-w-7xl mx-auto">
      <div className="grid lg:grid-cols-2 gap-8">
        {/* Controls Panel */}
        <Card className="p-6 space-y-6 border-border border !drop-shadow-xl">
          <div className="space-y-2">
            <h2 className="text-2xl font-bold text-white">Exoplanet Candidate Analysis</h2>
            <p className="text-sm text-gray-300">Enter observational data to generate a simulation and run AI analysis.</p>
          </div>

          {/* Input Fields */}
          <fieldset disabled={isProcessing} className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {(Object.keys(parameterLabels) as Array<keyof PlanetAnalysisParameters>).map((key) => (
                  <div className="space-y-2" key={key}>
                    <div className="flex items-center space-x-2">
                      <Label className="!text-white" htmlFor={key}>{parameterLabels[key].label}</Label>
                      <TooltipProvider>
                        <Tooltip>
                          <TooltipTrigger asChild>
                            <Info className="w-3 h-3 text-gray-300 cursor-pointer" />
                          </TooltipTrigger>
                          <TooltipContent>
                            <p>{parameterLabels[key].tooltip}</p>
                          </TooltipContent>
                        </Tooltip>
                      </TooltipProvider>
                    </div>
                    <Input
                      id={key}
                      type="text"
                      value={analysisParams[key]}
                      onChange={(e) => updateParameter(key, e.target.value)}
                      placeholder="e.g. 1.0"
                      className="!text-white"
                    />
                  </div>
                ))}
            </div>
          </fieldset>

          {/* Analyze Button */}
          <Button onClick={handleAnalyze} disabled={isProcessing} className="w-full" size="lg">
            {isProcessing ? (
              <>
                <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin mr-2" />
                Processing...
              </>
            ) : (
              <>
                <Sparkles className="w-4 h-4 mr-2" />
                Analyze & Simulate Planet
              </>
            )}
          </Button>

          {/* AI Analysis Result */}
          {analysisResult && (
            <Card className="p-4 bg-secondary/50 border-border">
              <h3 className="font-semibold text-lg mb-2">AI Analysis Result</h3>
              <div className="space-y-2 text-sm">
                <p>
                  <strong>Prediction:</strong>{" "}
                  <span
                    className={`font-bold ${
                      analysisResult.prediction === "Exoplanet" ? "text-green-500" : "text-red-500"
                    }`}
                  >
                    {analysisResult.prediction}
                  </span>
                </p>
                <p>
                  <strong>Confidence:</strong> {(analysisResult.confidence * 100).toFixed(2)}%
                </p>
                <p className="text-xs text-gray-300 pt-2">
                  This analysis is based on a machine learning model trained on confirmed exoplanet data.
                </p>
              </div>
            </Card>
          )}

          {/* AI Generation Prompt */}
          {generatedTexture && (
            <Card className="p-4 bg-accent/10 border-accent/20">
              <h4 className="text-sm font-semibold mb-2">AI Generation Prompt:</h4>
              <p className="text-xs text-gray-300 leading-relaxed">{generatedTexture.prompt}</p>
            </Card>
          )}
        </Card>

        {/* Preview Panel */}
        <div className="space-y-6">
          <Card className="p-6 border-border border">
            <h3 className="text-xl text-white font-bold mb-4">3D Preview</h3>
            <PlanetPreview parameters={simulationParams} isGenerating={isProcessing} textureUrl={generatedTexture?.url} />
          </Card>

          <Card className="p-6 border-border border">
            <h3 className="text-xl font-bold mb-4">Transit Light Curve</h3>
            <p className="text-sm text-gray-300 mb-4">
              This shows how the star's brightness would change if your planet passed in front of it
            </p>
            <LightCurveSimulation parameters={simulationParams} />
          </Card>
        </div>
      </div>
    </div>
  )
}