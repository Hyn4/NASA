'use client'

import { useState, useMemo, useEffect } from "react"
import { Card } from "@/components/ui/card"
import { Label } from "@/components/ui/label"
import { Button } from "@/components/ui/button"
import { Slider } from "@/components/ui/slider"
import { Earth, TrendingUp, Droplets, Thermometer, Mountain, RotateCcw, Info, AlertTriangle } from "lucide-react"
import { Alert, AlertDescription } from "@/components/ui/alert"
import type { PlanetData } from "@/app/laboratory/page"

interface EarthSimilarityCalculatorProps {
    initialData?: PlanetData | null
    onClear?: () => void
}

export function EarthSimilarityCalculator({ initialData, onClear }: EarthSimilarityCalculatorProps) {
    const [params, setParams] = useState({
        radius: 1.0,
        mass: 1.0,
        temperature: 288,
        escape_velocity: 11.2,
    })

    const [loadedPlanetName, setLoadedPlanetName] = useState<string | null>(null)
    const [sliderRanges, setSliderRanges] = useState({
        radius: { min: 0.1, max: 30.0 },
        mass: { min: 0.01, max: 100.0 },
        temperature: { min: 50, max: 3000 },
        escape_velocity: { min: 0.5, max: 100.0 }
    })
    const [outOfRangeWarnings, setOutOfRangeWarnings] = useState<string[]>([])

    // Carregar dados iniciais quando recebidos
    useEffect(() => {
        if (initialData) {
            const warnings: string[] = []
            const newRanges = { ...sliderRanges }

            // Validar e ajustar ranges se necessário
            let radius = initialData.radius || 1.0
            if (radius > sliderRanges.radius.max) {
                newRanges.radius.max = Math.ceil(radius * 1.2)
                warnings.push(`Radius adjusted to ${radius.toFixed(2)} R⊕ (beyond typical range)`)
            }
            if (radius < sliderRanges.radius.min) {
                newRanges.radius.min = Math.floor(radius * 0.8)
            }

            let mass = initialData.mass || 1.0
            if (mass > sliderRanges.mass.max) {
                newRanges.mass.max = Math.ceil(mass * 1.2)
                warnings.push(`Mass adjusted to ${mass.toFixed(2)} M⊕ (beyond typical range)`)
            }
            if (mass < sliderRanges.mass.min) {
                newRanges.mass.min = Math.floor(mass * 0.8 * 100) / 100
            }

            let temperature = initialData.temperature || 288
            if (temperature > sliderRanges.temperature.max) {
                newRanges.temperature.max = Math.ceil(temperature * 1.2)
                warnings.push(`Temperature adjusted to ${temperature.toFixed(0)}K (extremely hot planet)`)
            }
            if (temperature < sliderRanges.temperature.min) {
                newRanges.temperature.min = Math.floor(temperature * 0.8)
            }

            let escapeVelocity = initialData.escape_velocity || 11.2
            if (escapeVelocity > sliderRanges.escape_velocity.max) {
                newRanges.escape_velocity.max = Math.ceil(escapeVelocity * 1.2)
                warnings.push(`Escape velocity adjusted to ${escapeVelocity.toFixed(1)} km/s (very massive planet)`)
            }
            if (escapeVelocity < sliderRanges.escape_velocity.min) {
                newRanges.escape_velocity.min = Math.floor(escapeVelocity * 0.8 * 10) / 10
            }

            setSliderRanges(newRanges)
            setOutOfRangeWarnings(warnings)

            setParams({
                radius: radius,
                mass: mass,
                temperature: temperature,
                escape_velocity: escapeVelocity,
            })
            setLoadedPlanetName(initialData.name)
        }
    }, [initialData])

    const handleReset = () => {
        setParams({
            radius: 1.0,
            mass: 1.0,
            temperature: 288,
            escape_velocity: 11.2,
        })
        setLoadedPlanetName(null)
        setOutOfRangeWarnings([])
        // Resetar ranges para padrão
        setSliderRanges({
            radius: { min: 0.1, max: 30.0 },
            mass: { min: 0.01, max: 100.0 },
            temperature: { min: 50, max: 3000 },
            escape_velocity: { min: 0.5, max: 100.0 }
        })
        onClear?.()
    }

    // Calcular Earth Similarity Index (ESI)
    const similarity = useMemo(() => {
        const EARTH = {
            radius: 1.0,
            mass: 1.0,
            temperature: 288,
            escape_velocity: 11.2
        }

        const weights = {
            radius: 0.57,
            escape_velocity: 0.26,
            temperature: 0.17
        }

        const radiusSimilarity = 1 - Math.abs((params.radius - EARTH.radius) / (params.radius + EARTH.radius))
        const densitySimilarity = 1 - Math.abs(
            (params.escape_velocity - EARTH.escape_velocity) / (params.escape_velocity + EARTH.escape_velocity)
        )
        const tempSimilarity = 1 - Math.abs((params.temperature - EARTH.temperature) / (params.temperature + EARTH.temperature))

        const esi = Math.pow(
            Math.pow(radiusSimilarity, weights.radius) *
            Math.pow(densitySimilarity, weights.escape_velocity) *
            Math.pow(tempSimilarity, weights.temperature),
            1 / (weights.radius + weights.escape_velocity + weights.temperature)
        )

        const density = (params.mass / Math.pow(params.radius, 3)) * 5.51

        let classification = ""
        let habitability = ""

        if (esi > 0.9) {
            classification = "Earth Twin"
            habitability = "Highly Habitable"
        } else if (esi > 0.8) {
            classification = "Earth-like"
            habitability = "Potentially Habitable"
        } else if (esi > 0.6) {
            classification = "Similar to Earth"
            habitability = "Marginally Habitable"
        } else if (esi > 0.4) {
            classification = "Somewhat Earth-like"
            habitability = "Low Habitability"
        } else {
            classification = "Not Earth-like"
            habitability = "Unlikely Habitable"
        }

        return {
            esi: esi * 100,
            density,
            classification,
            habitability,
            individual: {
                radius: radiusSimilarity * 100,
                density: densitySimilarity * 100,
                temperature: tempSimilarity * 100
            }
        }
    }, [params])

    const updateParam = (key: keyof typeof params, value: number) => {
        setParams(prev => ({ ...prev, [key]: value }))
    }

    return (
        <div className="space-y-6">
            <div className="flex items-center justify-between">
                <div className="space-y-2">
                    <h2 className="text-2xl font-bold text-white">Earth Similarity Index (ESI)</h2>
                    <p className="text-gray-400">
                        Calculate how similar an exoplanet is to Earth based on physical properties
                    </p>
                </div>
                {loadedPlanetName && (
                    <Button
                        onClick={handleReset}
                        variant="outline"
                        className="bg-[#2c2c2c] border-[#404040] hover:bg-[#3c3c3c]"
                    >
                        <RotateCcw className="w-4 h-4 mr-2" />
                        Reset to Earth
                    </Button>
                )}
            </div>

            {/* Alert quando planeta está carregado */}
            {loadedPlanetName && (
                <Alert className="bg-blue-600/20 border-blue-500/30">
                    <Info className="h-4 w-4 text-blue-400" />
                    <AlertDescription className="text-blue-200">
                        <strong>Analyzing:</strong> {loadedPlanetName} - Data loaded from NASA Exoplanet Archive
                    </AlertDescription>
                </Alert>
            )}

            {/* Warnings para valores fora do range típico */}
            {outOfRangeWarnings.length > 0 && (
                <Alert className="bg-yellow-600/20 border-yellow-500/30">
                    <AlertTriangle className="h-4 w-4 text-yellow-400" />
                    <AlertDescription className="text-yellow-200">
                        <strong>Extreme Values Detected:</strong>
                        <ul className="mt-2 ml-4 space-y-1 text-xs">
                            {outOfRangeWarnings.map((warning, idx) => (
                                <li key={idx}>• {warning}</li>
                            ))}
                        </ul>
                    </AlertDescription>
                </Alert>
            )}

            {initialData && initialData.orbital_period && (
                <Card className="p-4 bg-[#2c2c2c] border-[#404040]">
                    <h3 className="text-sm font-bold text-white mb-3">Additional Planet Information</h3>
                    <div className="grid grid-cols-3 gap-4 text-xs">
                        {initialData.orbital_period && (
                            <div>
                                <p className="text-gray-500 mb-1">Orbital Period</p>
                                <p className="text-white font-mono">{initialData.orbital_period.toFixed(2)} days</p>
                            </div>
                        )}
                        {initialData.distance_from_star && (
                            <div>
                                <p className="text-gray-500 mb-1">Distance from Star</p>
                                <p className="text-white font-mono">{initialData.distance_from_star.toFixed(3)} AU</p>
                            </div>
                        )}
                        {initialData.density && (
                            <div>
                                <p className="text-gray-500 mb-1">Known Density</p>
                                <p className="text-white font-mono">{initialData.density.toFixed(2)} g/cm³</p>
                            </div>
                        )}
                    </div>
                </Card>
            )}

            <div className="grid lg:grid-cols-2 gap-6">
                {/* Input Parameters */}
                <div className="space-y-6">
                    <Card className="p-6 bg-[#2c2c2c] border-[#404040]">
                        <h3 className="text-lg font-bold text-white mb-4">Planet Parameters</h3>

                        <div className="space-y-5">
                            <div className="space-y-3">
                                <div className="flex items-center justify-between">
                                    <Label className="text-white flex items-center gap-2">
                                        <Mountain className="w-4 h-4 text-blue-400" />
                                        Planet Radius
                                    </Label>
                                    <span className="text-sm font-mono text-gray-400">{params.radius.toFixed(2)} R⊕</span>
                                </div>
                                <Slider
                                    value={[params.radius]}
                                    onValueChange={(v) => updateParam('radius', v[0])}
                                    min={sliderRanges.radius.min}
                                    max={sliderRanges.radius.max}
                                    step={0.01}
                                    className="w-full"
                                />
                                <div className="flex justify-between text-xs text-gray-500">
                                    <span>{sliderRanges.radius.min} R⊕</span>
                                    <span>Earth radii (1.0 = Earth)</span>
                                    <span>{sliderRanges.radius.max} R⊕</span>
                                </div>
                            </div>

                            <div className="space-y-3">
                                <div className="flex items-center justify-between">
                                    <Label className="text-white flex items-center gap-2">
                                        <TrendingUp className="w-4 h-4 text-purple-400" />
                                        Planet Mass
                                    </Label>
                                    <span className="text-sm font-mono text-gray-400">{params.mass.toFixed(2)} M⊕</span>
                                </div>
                                <Slider
                                    value={[params.mass]}
                                    onValueChange={(v) => updateParam('mass', v[0])}
                                    min={sliderRanges.mass.min}
                                    max={sliderRanges.mass.max}
                                    step={0.01}
                                    className="w-full"
                                />
                                <div className="flex justify-between text-xs text-gray-500">
                                    <span>{sliderRanges.mass.min} M⊕</span>
                                    <span>Earth masses (1.0 = Earth)</span>
                                    <span>{sliderRanges.mass.max} M⊕</span>
                                </div>
                            </div>

                            <div className="space-y-3">
                                <div className="flex items-center justify-between">
                                    <Label className="text-white flex items-center gap-2">
                                        <Thermometer className="w-4 h-4 text-red-400" />
                                        Surface Temperature
                                    </Label>
                                    <span className="text-sm font-mono text-gray-400">{params.temperature}K</span>
                                </div>
                                <Slider
                                    value={[params.temperature]}
                                    onValueChange={(v) => updateParam('temperature', v[0])}
                                    min={sliderRanges.temperature.min}
                                    max={sliderRanges.temperature.max}
                                    step={1}
                                    className="w-full"
                                />
                                <div className="flex justify-between text-xs text-gray-500">
                                    <span>{sliderRanges.temperature.min}K</span>
                                    <span>288K = Earth, 273K = water freezing</span>
                                    <span>{sliderRanges.temperature.max}K</span>
                                </div>
                            </div>

                            <div className="space-y-3">
                                <div className="flex items-center justify-between">
                                    <Label className="text-white flex items-center gap-2">
                                        <Droplets className="w-4 h-4 text-cyan-400" />
                                        Escape Velocity
                                    </Label>
                                    <span className="text-sm font-mono text-gray-400">{params.escape_velocity.toFixed(1)} km/s</span>
                                </div>
                                <Slider
                                    value={[params.escape_velocity]}
                                    onValueChange={(v) => updateParam('escape_velocity', v[0])}
                                    min={sliderRanges.escape_velocity.min}
                                    max={sliderRanges.escape_velocity.max}
                                    step={0.1}
                                    className="w-full"
                                />
                                <div className="flex justify-between text-xs text-gray-500">
                                    <span>{sliderRanges.escape_velocity.min} km/s</span>
                                    <span>11.2 km/s = Earth escape velocity</span>
                                    <span>{sliderRanges.escape_velocity.max} km/s</span>
                                </div>
                            </div>
                        </div>
                    </Card>

                    <Card className="p-6 bg-[#2c2c2c] border-[#404040]">
                        <h3 className="text-lg font-bold text-white mb-4">Derived Properties</h3>
                        <div className="space-y-3">
                            <div className="flex justify-between items-center">
                                <span className="text-gray-400">Density:</span>
                                <span className="text-white font-mono">{similarity.density.toFixed(2)} g/cm³</span>
                            </div>
                            <div className="flex justify-between items-center">
                                <span className="text-gray-400">Classification:</span>
                                <span className="text-blue-400 font-semibold">{similarity.classification}</span>
                            </div>
                            <div className="flex justify-between items-center">
                                <span className="text-gray-400">Habitability:</span>
                                <span className={`font-semibold ${similarity.esi > 80 ? 'text-green-400' :
                                    similarity.esi > 60 ? 'text-yellow-400' : 'text-red-400'
                                    }`}>
                                    {similarity.habitability}
                                </span>
                            </div>
                        </div>
                    </Card>
                </div>

                {/* Results */}
                <div className="space-y-6">
                    <Card className="p-8 bg-gradient-to-br from-blue-600/20 to-green-600/20 border-blue-500/30 text-center">
                        <div className="flex justify-center mb-4">
                            <div className="p-4 bg-white/10 rounded-full">
                                <Earth className="w-12 h-12 text-white" />
                            </div>
                        </div>
                        <h3 className="text-sm text-gray-300 mb-2">Earth Similarity Index</h3>
                        <div className="text-6xl font-bold text-white mb-2">
                            {similarity.esi.toFixed(1)}%
                        </div>
                        <p className="text-sm text-gray-300">
                            {similarity.esi > 90 ? "Almost identical to Earth!" :
                                similarity.esi > 70 ? "Very similar to Earth" :
                                    similarity.esi > 50 ? "Moderately Earth-like" :
                                        "Significantly different from Earth"}
                        </p>
                    </Card>

                    <Card className="p-6 bg-[#2c2c2c] border-[#404040]">
                        <h3 className="text-lg font-bold text-white mb-4">Component Similarities</h3>
                        <div className="space-y-4">
                            <div>
                                <div className="flex justify-between items-center mb-2">
                                    <span className="text-gray-400 text-sm">Radius Similarity</span>
                                    <span className="text-white font-mono text-sm">{similarity.individual.radius.toFixed(1)}%</span>
                                </div>
                                <div className="w-full bg-[#1b1b1b] rounded-full h-2">
                                    <div
                                        className="bg-gradient-to-r from-blue-500 to-cyan-500 h-2 rounded-full transition-all duration-500"
                                        style={{ width: `${Math.min(similarity.individual.radius, 100)}%` }}
                                    />
                                </div>
                            </div>

                            <div>
                                <div className="flex justify-between items-center mb-2">
                                    <span className="text-gray-400 text-sm">Density Similarity</span>
                                    <span className="text-white font-mono text-sm">{similarity.individual.density.toFixed(1)}%</span>
                                </div>
                                <div className="w-full bg-[#1b1b1b] rounded-full h-2">
                                    <div
                                        className="bg-gradient-to-r from-purple-500 to-pink-500 h-2 rounded-full transition-all duration-500"
                                        style={{ width: `${Math.min(similarity.individual.density, 100)}%` }}
                                    />
                                </div>
                            </div>

                            <div>
                                <div className="flex justify-between items-center mb-2">
                                    <span className="text-gray-400 text-sm">Temperature Similarity</span>
                                    <span className="text-white font-mono text-sm">{similarity.individual.temperature.toFixed(1)}%</span>
                                </div>
                                <div className="w-full bg-[#1b1b1b] rounded-full h-2">
                                    <div
                                        className="bg-gradient-to-r from-orange-500 to-red-500 h-2 rounded-full transition-all duration-500"
                                        style={{ width: `${Math.min(similarity.individual.temperature, 100)}%` }}
                                    />
                                </div>
                            </div>
                        </div>
                    </Card>

                    <Card className="p-4 bg-[#2c2c2c]/50 border-[#404040]">
                        <p className="text-xs text-gray-400 leading-relaxed">
                            <strong className="text-white">ESI Formula:</strong> The Earth Similarity Index uses a weighted geometric mean
                            of radius (57%), escape velocity/density (26%), and temperature (17%) similarities.
                            A score of 1.0 (100%) indicates Earth-twin conditions. The formula was developed by Schulze-Makuch et al. (2011).
                        </p>
                    </Card>
                </div>
            </div>
        </div>
    )
}
