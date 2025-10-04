// lib/nasa-api.ts
interface PlanetTextureParams {
  radius: number
  mass: number
  temperature: number
  starType: "red-dwarf" | "sun-like" | "blue-giant"
  distance: number
  planetType: string
}

export async function generatePlanetTexture(params: PlanetTextureParams) {
  try {
    const response = await fetch('/api/generate-texture', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(params),
    })

    if (!response.ok) {
      throw new Error('Failed to generate texture')
    }

    const data = await response.json()
    return {
      textureUrl: data.imageUrl,
      prompt: data.prompt,
    }
  } catch (error) {
    console.error('Error generating planet texture:', error)
    throw error
  }
}
