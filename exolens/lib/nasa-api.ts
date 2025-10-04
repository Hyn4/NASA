export interface ExoplanetData {
  name: string
  radius: number
  mass: number
  period: number
  distance: number
  temperature: number
  discovered: string
  hostStar: string
  starRadius: number
  starTemp: number
}

export interface SystemData {
  system: string
  planets: ExoplanetData[]
  count: number
}

export async function fetchExoplanetSystem(systemName = "TRAPPIST-1"): Promise<SystemData> {
  try {
    const response = await fetch(`/api/nasa/exoplanets?system=${encodeURIComponent(systemName)}`)

    if (!response.ok) {
      throw new Error(`Failed to fetch exoplanet data: ${response.statusText}`)
    }

    const data = await response.json()
    return data
  } catch (error) {
    console.error("Error fetching exoplanet system:", error)
    throw error
  }
}

export async function generatePlanetTexture(params: {
  radius: number
  temperature: number
  mass: number
  starType: string
  planetType: string
}): Promise<{ textureUrl: string; prompt: string }> {
  try {
    const response = await fetch("/api/generate-texture", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(params),
    })

    if (!response.ok) {
      throw new Error(`Failed to generate texture: ${response.statusText}`)
    }

    const data = await response.json()
    return {
      textureUrl: data.textureUrl,
      prompt: data.prompt,
    }
  } catch (error) {
    console.error("Error generating planet texture:", error)
    throw error
  }
}
