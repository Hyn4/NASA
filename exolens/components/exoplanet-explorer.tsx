'use client'

import { useState, useEffect } from "react"
import { Input } from "@/components/ui/input"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Search, Star, MapPin, CheckCircle2, AlertCircle, ChevronDown, ChevronRight, Loader2, Sparkles, Thermometer, Globe, Calculator } from "lucide-react"
import { useToast } from "@/hooks/use-toast"
import type { PlanetData } from "@/app/laboratory/page"

interface ExoplanetData {
    pl_name: string
    hostname: string
    discoverymethod: string
    disc_year: number
    disc_facility: string
    pl_rade: number
    pl_masse: number
    pl_orbper: number
    pl_orbsmax: number
    pl_eqt: number
    pl_insol: number
    pl_dens: number
    st_spectype: string
    st_teff: number
    st_rad: number
    st_mass: number
    st_logg: number
    sy_dist: number
    sy_vmag: number
    default_flag: number
}

interface PopularSystem {
    hostname: string
    planet_count: number
    st_spectype: string
    st_teff: number
    sy_dist: number
}

interface ExoplanetExplorerProps {
    onPlanetSelect: (planetData: PlanetData) => void
}

export function ExoplanetExplorer({ onPlanetSelect }: ExoplanetExplorerProps) {
    const [searchQuery, setSearchQuery] = useState("")
    const [searchType, setSearchType] = useState<"star" | "planet">("star")
    const [isLoading, setIsLoading] = useState(false)
    const [stars, setStars] = useState<{ [key: string]: ExoplanetData[] }>({})
    const [expandedStars, setExpandedStars] = useState<Set<string>>(new Set())
    const [popularSystems, setPopularSystems] = useState<PopularSystem[]>([])
    const { toast } = useToast()

    useEffect(() => {
        loadPopularSystems()
    }, [])

    const loadPopularSystems = async () => {
        try {
            const response = await fetch('/api/nasa/popular-systems')
            if (response.ok) {
                const data = await response.json()
                setPopularSystems(data.systems)
            }
        } catch (error) {
            console.error('Error loading popular systems:', error)
        }
    }

    const handleSearch = async () => {
        if (!searchQuery.trim()) {
            toast({
                title: "Search Required",
                description: "Please enter a star name or exoplanet name to search.",
                variant: "destructive"
            })
            return
        }

        setIsLoading(true)

        try {
            const response = await fetch(
                `/api/nasa/search-exoplanets?q=${encodeURIComponent(searchQuery)}&type=${searchType}`
            )

            if (!response.ok) {
                throw new Error('Failed to fetch data from NASA')
            }

            const data = await response.json()

            if (data.count === 0) {
                toast({
                    title: "No Results",
                    description: "No exoplanets or stars found matching your search.",
                    variant: "destructive"
                })
                setStars({})
                return
            }

            setStars(data.results)
            toast({
                title: "Search Complete",
                description: `Found ${data.stars} star system(s) with ${data.count} exoplanet(s).`
            })

        } catch (error: any) {
            console.error('Search error:', error)
            toast({
                title: "Search Failed",
                description: error.message || "Unable to fetch exoplanet data.",
                variant: "destructive"
            })
        } finally {
            setIsLoading(false)
        }
    }

    const handleQuickSearch = (systemName: string) => {
        setSearchQuery(systemName)
        setSearchType("star")
        setTimeout(() => {
            const button = document.querySelector('[data-search-button]') as HTMLButtonElement
            button?.click()
        }, 100)
    }

    const toggleStar = (hostname: string) => {
        const newExpanded = new Set(expandedStars)
        if (newExpanded.has(hostname)) {
            newExpanded.delete(hostname)
        } else {
            newExpanded.add(hostname)
        }
        setExpandedStars(newExpanded)
    }

    const handleAnalyzePlanet = (planet: ExoplanetData) => {
        // Calcular escape velocity baseado em massa e raio
        // v_esc = sqrt(2 * G * M / R)
        // Onde G = 6.674×10^-11 m^3 kg^-1 s^-2
        const G = 6.674e-11
        const earthMass = 5.972e24 // kg
        const earthRadius = 6.371e6 // metros

        const planetMassKg = (planet.pl_masse || 1) * earthMass
        const planetRadiusM = (planet.pl_rade || 1) * earthRadius

        const escapeVelocity = Math.sqrt(2 * G * planetMassKg / planetRadiusM) / 1000 // converter para km/s

        const planetData: PlanetData = {
            name: planet.pl_name,
            radius: planet.pl_rade || 1.0,
            mass: planet.pl_masse || 1.0,
            temperature: planet.pl_eqt || 288,
            escape_velocity: escapeVelocity,
            density: planet.pl_dens,
            orbital_period: planet.pl_orbper,
            distance_from_star: planet.pl_orbsmax
        }

        onPlanetSelect(planetData)

        toast({
            title: "Planet Loaded",
            description: `${planet.pl_name} data loaded into Earth Similarity Calculator`,
        })
    }

    const getStarType = (specType: string | null) => {
        if (!specType) return { name: "Unknown", color: "gray" }
        const type = specType.charAt(0).toUpperCase()
        const typeMap: { [key: string]: { name: string, color: string } } = {
            'O': { name: 'Blue Giant', color: 'blue' },
            'B': { name: 'Blue Star', color: 'blue' },
            'A': { name: 'Blue-White', color: 'cyan' },
            'F': { name: 'White Star', color: 'white' },
            'G': { name: 'Yellow (Sun-like)', color: 'yellow' },
            'K': { name: 'Orange Star', color: 'orange' },
            'M': { name: 'Red Dwarf', color: 'red' }
        }
        return typeMap[type] || { name: specType, color: 'gray' }
    }

    const getHabitabilityScore = (planet: ExoplanetData) => {
        if (!planet.pl_eqt || !planet.pl_rade) return null

        const tempScore = planet.pl_eqt >= 273 && planet.pl_eqt <= 373 ? 1 : 0
        const radiusScore = planet.pl_rade >= 0.5 && planet.pl_rade <= 2.0 ? 1 : 0
        const totalScore = (tempScore + radiusScore) / 2

        if (totalScore >= 0.8) return { label: "High", color: "green" }
        if (totalScore >= 0.4) return { label: "Medium", color: "yellow" }
        return { label: "Low", color: "red" }
    }

    return (
        <div className="space-y-6">
            {/* Search Bar */}
            <div className="space-y-3">
                <div className="flex gap-3">
                    <div className="relative flex-1">
                        <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-5 h-5" />
                        <Input
                            placeholder={searchType === "star"
                                ? "Search by star name (e.g., Kepler-186, TRAPPIST-1, Proxima Cen)..."
                                : "Search by exoplanet name (e.g., Kepler-186 f, TRAPPIST-1 e)..."}
                            value={searchQuery}
                            onChange={(e) => setSearchQuery(e.target.value)}
                            onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
                            className="pl-10 bg-[#2c2c2c] border-[#404040] text-white placeholder:text-gray-500"
                        />
                    </div>
                    <Button
                        onClick={handleSearch}
                        disabled={isLoading}
                        data-search-button
                        className="bg-blue-600 hover:bg-blue-700 px-8"
                    >
                        {isLoading ? (
                            <>
                                <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                                Searching...
                            </>
                        ) : (
                            'Search'
                        )}
                    </Button>
                </div>

                <div className="flex gap-2">
                    <Button
                        variant={searchType === "star" ? "default" : "outline"}
                        size="sm"
                        onClick={() => setSearchType("star")}
                        className={searchType === "star" ? "bg-blue-600" : "bg-[#2c2c2c] border-[#404040]"}
                    >
                        <Star className="w-4 h-4 mr-2" />
                        Search Stars
                    </Button>
                    <Button
                        variant={searchType === "planet" ? "default" : "outline"}
                        size="sm"
                        onClick={() => setSearchType("planet")}
                        className={searchType === "planet" ? "bg-blue-600" : "bg-[#2c2c2c] border-[#404040]"}
                    >
                        <Globe className="w-4 h-4 mr-2" />
                        Search Planets
                    </Button>
                </div>
            </div>

            {/* Popular Systems */}
            {Object.keys(stars).length === 0 && popularSystems.length > 0 && (
                <Card className="p-6 bg-[#2c2c2c] border-[#404040]">
                    <h3 className="text-lg font-bold text-white mb-4 flex items-center gap-2">
                        <Sparkles className="w-5 h-5 text-yellow-400" />
                        Popular Star Systems
                    </h3>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                        {popularSystems.map((system) => (
                            <button
                                key={system.hostname}
                                onClick={() => handleQuickSearch(system.hostname)}
                                className="p-3 bg-[#1b1b1b] rounded-lg border border-[#404040] hover:border-blue-500 transition-all text-left"
                            >
                                <p className="text-white font-semibold text-sm mb-1">{system.hostname}</p>
                                <p className="text-xs text-gray-400">{system.planet_count} planet(s)</p>
                                {system.sy_dist && (
                                    <p className="text-xs text-blue-400 mt-1">{system.sy_dist.toFixed(1)} pc</p>
                                )}
                            </button>
                        ))}
                    </div>
                </Card>
            )}

            {/* Results */}
            {Object.keys(stars).length > 0 ? (
                <div className="space-y-4">
                    <div className="flex items-center justify-between">
                        <h3 className="text-xl font-bold text-white">
                            Found {Object.keys(stars).length} Star System(s)
                        </h3>
                    </div>

                    <div className="space-y-3">
                        {Object.entries(stars).map(([hostname, planets]) => {
                            const isExpanded = expandedStars.has(hostname)
                            const firstPlanet = planets[0]
                            const starType = getStarType(firstPlanet.st_spectype)

                            return (
                                <Card key={hostname} className="bg-[#2c2c2c] border-[#404040] overflow-hidden">
                                    <button
                                        onClick={() => toggleStar(hostname)}
                                        className="w-full p-5 flex items-center justify-between hover:bg-[#333333] transition-colors"
                                    >
                                        <div className="flex items-center gap-4">
                                            <div className={`p-3 rounded-lg bg-${starType.color}-500/20`}>
                                                <Star className={`w-6 h-6 text-${starType.color}-400`} />
                                            </div>
                                            <div className="text-left">
                                                <h4 className="text-xl font-bold text-white">{hostname}</h4>
                                                <div className="flex items-center gap-4 mt-2 text-sm">
                                                    <Badge variant="outline" className="text-gray-300 border-[#404040]">
                                                        {starType.name}
                                                    </Badge>
                                                    {firstPlanet.sy_dist && (
                                                        <span className="flex items-center gap-1 text-gray-400">
                                                            <MapPin className="w-3 h-3" />
                                                            {firstPlanet.sy_dist.toFixed(2)} parsecs
                                                        </span>
                                                    )}
                                                    {firstPlanet.st_teff && (
                                                        <span className="flex items-center gap-1 text-orange-400">
                                                            <Thermometer className="w-3 h-3" />
                                                            {firstPlanet.st_teff.toFixed(0)}K
                                                        </span>
                                                    )}
                                                    <Badge className="bg-blue-600 text-white">
                                                        {planets.length} planet(s)
                                                    </Badge>
                                                </div>
                                            </div>
                                        </div>
                                        <div className="text-gray-400">
                                            {isExpanded ? <ChevronDown className="w-6 h-6" /> : <ChevronRight className="w-6 h-6" />}
                                        </div>
                                    </button>

                                    {isExpanded && (
                                        <div className="border-t border-[#404040] bg-[#1b1b1b]">
                                            <div className="p-5 space-y-3">
                                                {planets.map((planet, idx) => {
                                                    const habitability = getHabitabilityScore(planet)

                                                    return (
                                                        <div
                                                            key={idx}
                                                            className="p-5 bg-[#2c2c2c] rounded-lg border border-[#404040] hover:border-blue-500/50 transition-colors"
                                                        >
                                                            <div className="flex items-start justify-between mb-4">
                                                                <div className="flex-1">
                                                                    <h5 className="text-white font-bold text-lg mb-1">{planet.pl_name}</h5>
                                                                    <div className="flex items-center gap-3 text-xs text-gray-400 mt-2">
                                                                        <span>Discovered: {planet.disc_year || 'Unknown'}</span>
                                                                        <span>•</span>
                                                                        <span>Method: {planet.discoverymethod || 'Unknown'}</span>
                                                                        {planet.disc_facility && (
                                                                            <>
                                                                                <span>•</span>
                                                                                <span>{planet.disc_facility}</span>
                                                                            </>
                                                                        )}
                                                                    </div>
                                                                </div>
                                                                <div className="flex flex-col items-end gap-2">
                                                                    {planet.default_flag === 1 ? (
                                                                        <Badge className="bg-green-600 text-white">
                                                                            <CheckCircle2 className="w-3 h-3 mr-1" />
                                                                            Confirmed
                                                                        </Badge>
                                                                    ) : (
                                                                        <Badge variant="outline" className="border-yellow-500 text-yellow-400">
                                                                            <AlertCircle className="w-3 h-3 mr-1" />
                                                                            Candidate
                                                                        </Badge>
                                                                    )}
                                                                    {habitability && (
                                                                        <Badge
                                                                            className={
                                                                                habitability.color === "green" ? "bg-green-600" :
                                                                                    habitability.color === "yellow" ? "bg-yellow-600" :
                                                                                        "bg-red-600"
                                                                            }
                                                                        >
                                                                            Habitability: {habitability.label}
                                                                        </Badge>
                                                                    )}
                                                                </div>
                                                            </div>

                                                            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm mb-4">
                                                                <div className="bg-[#1b1b1b] p-3 rounded">
                                                                    <p className="text-gray-500 text-xs mb-1">Radius</p>
                                                                    <p className="text-white font-mono text-base">
                                                                        {planet.pl_rade ? `${planet.pl_rade.toFixed(2)} R⊕` : 'N/A'}
                                                                    </p>
                                                                </div>
                                                                <div className="bg-[#1b1b1b] p-3 rounded">
                                                                    <p className="text-gray-500 text-xs mb-1">Mass</p>
                                                                    <p className="text-white font-mono text-base">
                                                                        {planet.pl_masse ? `${planet.pl_masse.toFixed(2)} M⊕` : 'N/A'}
                                                                    </p>
                                                                </div>
                                                                <div className="bg-[#1b1b1b] p-3 rounded">
                                                                    <p className="text-gray-500 text-xs mb-1">Orbital Period</p>
                                                                    <p className="text-white font-mono text-base">
                                                                        {planet.pl_orbper ? `${planet.pl_orbper.toFixed(2)} days` : 'N/A'}
                                                                    </p>
                                                                </div>
                                                                <div className="bg-[#1b1b1b] p-3 rounded">
                                                                    <p className="text-gray-500 text-xs mb-1">Temperature</p>
                                                                    <p className="text-white font-mono text-base">
                                                                        {planet.pl_eqt ? `${planet.pl_eqt.toFixed(0)}K` : 'N/A'}
                                                                    </p>
                                                                </div>
                                                            </div>

                                                            {/* Botão de Análise */}
                                                            <Button
                                                                onClick={() => handleAnalyzePlanet(planet)}
                                                                className="w-full bg-gradient-to-r from-green-600 to-emerald-600 hover:from-green-700 hover:to-emerald-700"
                                                            >
                                                                <Calculator className="w-4 h-4 mr-2" />
                                                                Analyze Earth Similarity
                                                            </Button>
                                                        </div>
                                                    )
                                                })}
                                            </div>
                                        </div>
                                    )}
                                </Card>
                            )
                        })}
                    </div>
                </div>
            ) : !isLoading && (
                <Card className="p-12 bg-[#2c2c2c] border-[#404040] text-center">
                    <div className="flex flex-col items-center gap-4">
                        <div className="p-6 bg-[#3c3c3c] rounded-full">
                            <Search className="w-12 h-12 text-gray-500" />
                        </div>
                        <div>
                            <h3 className="text-xl font-bold text-white mb-2">Start Exploring</h3>
                            <p className="text-gray-400 max-w-md">
                                Search for stars or exoplanets from the NASA Exoplanet Archive.
                                Try the popular systems above or search for "Kepler", "TRAPPIST", "Proxima", or "TOI".
                            </p>
                        </div>
                    </div>
                </Card>
            )}
        </div>
    )
}
