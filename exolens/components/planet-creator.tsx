'use client'

import { useState, useMemo, useEffect } from "react"
import { Card } from "@/components/ui/card"
import { Label } from "@/components/ui/label"
import { Input } from "@/components/ui/input"
import { Button } from "@/components/ui/button"
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group"
import { PlanetPreview } from "@/components/planet-preview"
import { LightCurveSimulation } from "@/components/light-curve-simulation"
import { Sparkles, Box, LineChart, Info } from "lucide-react"
import { generatePlanetTexture } from "@/lib/nasa-api"
import { useToast } from "@/hooks/use-toast"
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip"

// Interface para os inputs do usuário
export interface PlanetAnalysisParameters {
  pl_orbper: string // Orbital Period (days)
  pl_rade: string // Planet Radius (Earth Radius)
  pl_trandep: string // Transit Depth (ppm)
  st_teff: string // Stellar Effective Temperature (K)
  st_rad: string // Stellar Radius (Solar Radius)
  st_logg: string // Stellar Surface Gravity (log10(cm/s**2))
}

// Interface para a simulação (PlanetPreview, LightCurve)
export interface PlanetSimulationParameters {
  radius: number
  mass: number
  temperature: number
  starType: "red-dwarf" | "sun-like" | "blue-giant"
  distance: number
}

type AiMode = 'feature' | 'lightcurve'

const parameterLabels: { [key in keyof PlanetAnalysisParameters]: { label: string; tooltip: string } } = {
  pl_orbper: { label: "Orbital Period (days)", tooltip: "Time the planet takes to orbit its star." },
  pl_rade: { label: "Planet Radius (R⊕)", tooltip: "The radius of the planet in multiples of Earth's radius." },
  pl_trandep: { label: "Transit Depth (ppm)", tooltip: "The percentage of the star's light blocked by the planet, in parts per million." },
  st_teff: { label: "Stellar Temperature (K)", tooltip: "The effective surface temperature of the host star in Kelvin." },
  st_rad: { label: "Stellar Radius (R☉)", tooltip: "The radius of the host star in multiples of the Sun's radius." },
  st_logg: { label: "Stellar Gravity (log(g))", tooltip: "The stellar surface gravity as a logarithm (base 10)." },
}

const starProperties = {
  "red-dwarf": { radius: 0.3, mass: 0.3, logg: 5.0 },
  "sun-like": { radius: 1.0, mass: 1.0, logg: 4.4 },
  "blue-giant": { radius: 10, mass: 15, logg: 3.5 },
}

const habitableZones = {
  "red-dwarf": { min: 0.1, max: 0.4 },
  "sun-like": { min: 0.8, max: 1.5 },
  "blue-giant": { min: 5, max: 25 },
}

export function PlanetCreator() {
  // State para os inputs do usuário (com Terra/Sol como padrão)
  const [analysisParams, setAnalysisParams] = useState<PlanetAnalysisParameters>({
    pl_orbper: "365.25",
    pl_rade: "1",
    pl_trandep: "84",
    st_teff: "5778",
    st_rad: "1",
    st_logg: "4.44",
  })

  // State para os parâmetros de simulação derivados
  const [simulationParams, setSimulationParams] = useState<PlanetSimulationParameters>({
    radius: 1,
    mass: 1,
    temperature: 5778,
    starType: "sun-like",
    distance: 1,
  })

  const [aiMode, setAiMode] = useState<AiMode>('feature')
  const [isProcessing, setIsProcessing] = useState(false)
  const [generatedTexture, setGeneratedTexture] = useState<{ url: string; prompt: string } | null>(null)
  const [analysisResult, setAnalysisResult] = useState<any>(null)
  const [isClient, setIsClient] = useState(false)
  const { toast } = useToast()

  useEffect(() => {
    setIsClient(true)
  }, [])

  // Este hook deriva os parâmetros de simulação dos parâmetros de análise
  useEffect(() => {
    const deriveSimulationParameters = () => {
      const pl_rade = parseFloat(analysisParams.pl_rade) || 0
      const st_teff = parseFloat(analysisParams.st_teff) || 0
      const st_rad = parseFloat(analysisParams.st_rad) || 0
      const st_logg = parseFloat(analysisParams.st_logg) || 0
      const pl_orbper = parseFloat(analysisParams.pl_orbper) || 0

      if (!pl_rade || !st_teff || !st_rad || !st_logg || !pl_orbper) {
        return // Dados insuficientes
      }

      // 1. Derivar tipo de estrela pela temperatura
      const starType: PlanetSimulationParameters["starType"] =
        st_teff < 4000 ? "red-dwarf" : st_teff < 7000 ? "sun-like" : "blue-giant"

      // 2. Derivar massa (aproximação: M proporcional a R^2)
      const mass = Math.pow(pl_rade, 2)

      // 3. Derivar distância do período orbital (3ª Lei de Kepler)
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
        temperature: st_teff,
        starType: starType,
        distance: isNaN(distance_au) ? 1 : distance_au,
      })
    }

    deriveSimulationParameters()
  }, [analysisParams])

  // Classificação do planeta baseada nos parâmetros
  const { planetType, density, isHabitable } = useMemo(() => {
    const { radius, mass, starType, distance } = simulationParams
    const type = radius < 1.6 ? "Terrestrial" : radius < 5 ? "Neptune-like" : "Gas Giant"
    const dens = (mass / Math.pow(radius, 3)) * 5.51
    const zone = habitableZones[starType]
    const habitable = distance >= zone.min && distance <= zone.max
    return {
      planetType: type,
      density: isNaN(dens) ? 0 : dens,
      isHabitable: habitable,
    }
  }, [simulationParams])

  const updateParameter = (key: keyof PlanetAnalysisParameters, value: string) => {
    if (/^[0-9]*\.?[0-9]*$/.test(value)) {
      setAnalysisParams((prev) => ({ ...prev, [key]: value }))
    }
  }

  const handleRunAnalysis = async () => {
    setIsProcessing(true)
    setGeneratedTexture(null)
    setAnalysisResult(null)

    const payload = Object.fromEntries(
      Object.entries(analysisParams).map(([key, value]) => [key, Number(value)])
    )

    if (Object.values(payload).some(v => isNaN(v) || v === 0)) {
      toast({ title: "Invalid Input", description: "Please ensure all fields are filled with valid numbers.", variant: "destructive" })
      setIsProcessing(false)
      return
    }

    try {
      const endpoint = aiMode === 'lightcurve' ? '/predict_lightcurve' : '/api/analyze-planet'

      const [textureResult, analysisResponse] = await Promise.all([
        generatePlanetTexture({
          ...simulationParams,
          planetType: planetType,
        }),
        fetch(endpoint, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        }),
      ])

      setGeneratedTexture({ url: textureResult.textureUrl, prompt: textureResult.prompt })
      toast({ title: "Planet Texture Generated!", description: "AI is creating a visual representation." })

      if (!analysisResponse.ok) {
        const errorData = await analysisResponse.json()
        throw new Error(errorData.error || `Analysis via ${endpoint} failed`)
      }
      const analysisData = await analysisResponse.json()
      setAnalysisResult(analysisData)
      toast({ title: "Analysis Complete!", description: `The ${aiMode === 'feature' ? 'Feature' : 'LightCurve'} AI has classified the exoplanet.` })

    } catch (error: any) {
      toast({ title: "An Error Occurred", description: error.message, variant: "destructive" })
    } finally {
      setIsProcessing(false)
    }
  }

  if (!isClient) {
    return <div className="max-w-7xl mx-auto"><p className="text-gray-400">Loading Laboratory...</p></div>
  }

  return (
    <div className="max-w-7xl mx-auto">
      <style jsx global>{`
        /* Estilo customizado para inputs dark theme */
        input[type="text"]:not([class*="text-"]) {
          color: #fafafa !important;
        }
      `}</style>
      <div className="grid lg:grid-cols-2 gap-8">
        {/* Painel de Controles */}
        <div className="bg-[#1b1b1b] text-gray-200 flex flex-col gap-6 rounded-xl border border-[#333333] shadow-sm p-6 space-y-6">
          <div className="space-y-2">
            <h2 className="text-2xl font-bold text-white">Exoplanet Parameters</h2>
            <p className="text-sm text-gray-400">Enter observational data to generate a simulation and run AI analysis.</p>
          </div>

          {/* Campos de Input */}
          <fieldset disabled={isProcessing} className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {(Object.keys(parameterLabels) as Array<keyof PlanetAnalysisParameters>).map((key) => (
                <div className="space-y-2" key={key}>
                  <div className="flex items-center space-x-2">
                    <Label htmlFor={key} className="text-white">{parameterLabels[key].label}</Label>
                    <TooltipProvider>
                      <Tooltip>
                        <TooltipTrigger asChild>
                          <Info className="w-3 h-3 text-gray-400 cursor-pointer hover:text-gray-300 transition-colors" />
                        </TooltipTrigger>
                        <TooltipContent className="bg-[#2c2c2c] border-[#404040]">
                          <p className="text-gray-300">{parameterLabels[key].tooltip}</p>
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
                    className="bg-[#2c2c2c] border-[#404040] text-white placeholder:text-gray-500 focus:border-blue-500 focus:ring-blue-500/20"
                  />
                </div>
              ))}
            </div>
          </fieldset>

          {/* Card de Classificação do Planeta */}
          <div className="text-gray-200 flex flex-col gap-3 rounded-xl border border-[#333333] shadow-sm p-4 bg-[#2c2c2c]/70">
            <h3 className="font-semibold text-sm text-white">Planet Classification</h3>
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-400">Type:</span>
                <span className="text-sm font-bold text-blue-400">{planetType}</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-400">Habitable Zone:</span>
                <span className={`text-sm font-bold ${isHabitable ? 'text-green-400' : 'text-red-400'}`}>
                  {isHabitable ? 'Yes' : 'No'}
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-400">Density:</span>
                <span className="text-sm font-mono text-white">{density.toFixed(2)} g/cm³</span>
              </div>
            </div>
          </div>

          {/* Toggle para selecionar o modelo de IA */}
          <ToggleGroup
            type="single"
            value={aiMode}
            onValueChange={(value: AiMode) => {
              if (value) setAiMode(value)
            }}
            className="grid grid-cols-2 gap-2"
            disabled={isProcessing}
          >
            <ToggleGroupItem
              value="feature"
              aria-label="Select Feature AI"
              className="py-6 text-lg justify-center data-[state=on]:bg-blue-600 data-[state=on]:text-white"
            >
              <Box className="w-5 h-5 mr-2" />
              Feature AI
            </ToggleGroupItem>
            <ToggleGroupItem
              value="lightcurve"
              aria-label="Select LightCurve AI"
              className="py-6 text-lg justify-center data-[state=on]:bg-blue-600 data-[state=on]:text-white"
            >
              <LineChart className="w-5 h-5 mr-2" />
              LightCurve AI
            </ToggleGroupItem>
          </ToggleGroup>

          {/* Botão de Análise */}
          <Button onClick={handleRunAnalysis} disabled={isProcessing} className="w-full bg-blue-600 hover:bg-blue-700 text-white">
            {isProcessing ? (
              <>
                <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin mr-2" />
                Processing...
              </>
            ) : (
              <>
                <Sparkles className="w-4 h-4 mr-2" />
                Run Analysis
              </>
            )}
          </Button>

          {/* Resultado da Análise de IA */}
          {analysisResult && (
            <Card className="p-4 bg-gray-900/70 border-[#333333]">
              <h3 className="font-semibold text-lg mb-2 text-white">AI Analysis Result</h3>
              <div className="space-y-2 text-sm">
                <p className="text-gray-300">
                  <strong>Prediction:</strong>{" "}
                  <span
                    className={`font-bold ${analysisResult.prediction === "Exoplanet" ? "text-green-400" : "text-red-400"}`}
                  >
                    {analysisResult.prediction}
                  </span>
                </p>
                <p className="text-gray-300">
                  <strong>Confidence:</strong> {(analysisResult.confidence * 100).toFixed(2)}%
                </p>
                <p className="text-xs text-gray-400 pt-2">
                  This analysis is based on a machine learning model trained on confirmed exoplanet data.
                </p>
              </div>
            </Card>
          )}

          {/* Prompt de Geração de IA */}
          {generatedTexture && (
            <Card className="p-4 bg-[#2c2c2c]/70 border-[#404040]">
              <h4 className="text-sm font-semibold mb-2 text-white">AI Generation Prompt:</h4>
              <p className="text-xs text-gray-400 leading-relaxed">{generatedTexture.prompt}</p>
            </Card>
          )}
        </div>

        {/* Painel de Preview */}
        <div className="space-y-6">
          <Card className="p-6 border-[#333333] border bg-[#1b1b1b]">
            <h3 className="text-xl text-white font-bold mb-4">3D Preview</h3>
            <PlanetPreview
              parameters={simulationParams}
              isGenerating={isProcessing}
              textureUrl={generatedTexture?.url}
            />
          </Card>

          <Card className="p-6 border-[#333333] border bg-[#1b1b1b]">
            <h3 className="text-xl font-bold text-white mb-4">Transit Light Curve</h3>
            <p className="text-sm text-gray-400 mb-4">
              This shows how the star's brightness would change if your planet passed in front of it
            </p>
            <LightCurveSimulation parameters={simulationParams} />
          </Card>
        </div>
      </div>
    </div>
  )
}
