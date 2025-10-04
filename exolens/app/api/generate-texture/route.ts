// app/api/generate-texture/route.ts
import { NextResponse } from "next/server"

export async function POST(request: Request) {
  try {
    const body = await request.json()
    const { radius, temperature, mass, starType, planetType } = body

    // Build a detailed prompt for AI image generation
    const prompt = buildTexturePrompt({ radius, temperature, mass, starType, planetType })

    const GOOGLE_API_KEY = process.env.GOOGLE_API_KEY

    if (!GOOGLE_API_KEY) {
      console.warn('Google API key not configured, returning placeholder')
      return NextResponse.json({
        success: true,
        prompt: prompt,
        textureUrl: `/placeholder.svg?height=1024&width=1024&query=${encodeURIComponent(prompt)}`,
        message: "Google API key not configured. Using placeholder image.",
      })
    }

    // Call Google Imagen API to generate the texture
    const imagenResponse = await fetch(
      `https://generativelanguage.googleapis.com/v1beta/models/imagen-3.0-generate-001:predict?key=${GOOGLE_API_KEY}`,
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          instances: [
            {
              prompt: prompt,
            },
          ],
          parameters: {
            sampleCount: 1,
            aspectRatio: "1:1",
            safetyFilterLevel: "block_some",
            personGeneration: "dont_allow"
          },
        }),
      }
    )

    if (!imagenResponse.ok) {
      const errorText = await imagenResponse.text()
      console.error('Imagen API Error:', errorText)
      
      // Fallback to placeholder if API fails
      return NextResponse.json({
        success: true,
        prompt: prompt,
        textureUrl: `/placeholder.svg?height=1024&width=1024&query=${encodeURIComponent(prompt)}`,
        message: "AI generation failed, using placeholder image.",
      })
    }

    const imagenData = await imagenResponse.json()
    
    // Extract base64 image from response
    const imageBase64 = imagenData.predictions[0].bytesBase64Encoded
    const imageUrl = `data:image/png;base64,${imageBase64}`

    return NextResponse.json({
      success: true,
      prompt: prompt,
      textureUrl: imageUrl,
      message: "AI-generated texture successfully created",
    })

  } catch (error) {
    console.error("Error generating texture:", error)
    return NextResponse.json(
      {
        error: "Failed to generate texture",
        message: error instanceof Error ? error.message : "Unknown error",
      },
      { status: 500 },
    )
  }
}

function buildTexturePrompt(params: {
  radius: number
  temperature: number
  mass: number
  starType: string
  planetType: string
}): string {
  const { radius, temperature, mass, starType, planetType } = params

  let baseDescription = ""
  let surfaceFeatures = ""
  let atmosphere = ""
  let lighting = ""
  let textureStyle = ""
  let gravity = mass / (radius * radius)

  // Determine texture style based on temperature
  let temperatureEffect = ""
  if (temperature > 1500) {
    temperatureEffect = "molten lava surface with glowing magma cracks and volcanic eruptions"
  } else if (temperature > 800) {
    temperatureEffect = "cracked basalt surface with volcanic features and dark lava flows"
  } else if (temperature > 400) {
    temperatureEffect = "dry rocky regolith with impact craters and barren landscape"
  } else if (temperature > 273) {
    temperatureEffect = "rocky surface with weathering patterns and mineral deposits"
  } else if (temperature > 200) {
    temperatureEffect = "partially frozen surface with ice formations and frozen lakes"
  } else {
    temperatureEffect = "completely frozen icy crust with methane ice and nitrogen frost"
  }

  // Determine planet characteristics based on type and parameters
  if (planetType === "Gas Giant") {
    baseDescription = "A massive gas giant planet"
    surfaceFeatures = `turbulent atmospheric storms, Jupiter-like colorful cloud bands with ${gravity > 2 ? 'fine-grained compressed' : 'large swirling'} storm systems`
    atmosphere = "thick hydrogen and helium atmosphere with dramatic storm vortices and color variations from ammonia and methane"
    textureStyle = "flat graphic style with bold banded patterns, cell-shaded cloud layers, vibrant colors"
  } else if (planetType === "Neptune-like") {
    baseDescription = "An ice giant planet"
    surfaceFeatures = `methane-rich clouds with subtle atmospheric bands and ${gravity > 1.5 ? 'compressed' : 'expanded'} atmospheric features`
    atmosphere = "thick atmosphere dominated by methane giving deep blue-green coloration with high-altitude white methane clouds"
    textureStyle = "flat graphic style with smooth gradients, blue-green palette, minimal surface detail"
  } else if (planetType === "Super-Earth") {
    if (temperature < 273) {
      baseDescription = "A large rocky super-Earth with frozen surface"
      surfaceFeatures = `frozen oceans covering most surface, massive ice sheets, ${gravity > 2 ? 'compressed and flat' : 'tall jagged'} ice formations, visible rocky continents through ice`
      atmosphere = "thin atmosphere with ice crystal hazes"
      textureStyle = "flat graphic style with white-blue ice textures, subtle shadows, clean edges"
    } else if (temperature < 373) {
      baseDescription = "A large Earth-like super-Earth with oceans"
      surfaceFeatures = `vast blue oceans, large continents with varied terrain, white cloud formations, ${gravity > 2 ? 'flat plains' : 'mountain ranges'}`
      atmosphere = "thick Earth-like atmosphere with water vapor clouds and dynamic weather systems"
      textureStyle = "flat graphic style with blue oceans, green-brown landmasses, white cloud wisps"
    } else {
      baseDescription = "A large hot super-Earth with volcanic surface"
      surfaceFeatures = `${temperatureEffect}, active volcanoes, extensive lava plains, minimal water`
      atmosphere = "thick atmosphere with volcanic ash plumes and heat shimmer"
      textureStyle = "flat graphic style with red-orange lava glow, dark rocky textures, dramatic lighting"
    }
  } else {
    // Terrestrial
    if (temperature < 200) {
      baseDescription = "A frozen terrestrial planet"
      surfaceFeatures = `${temperatureEffect}, completely frozen surface with no liquid water, nitrogen and methane ice formations`
      atmosphere = "very thin atmosphere with frozen gases, minimal cloud cover"
      textureStyle = "flat graphic style with pale blue-white ice, clean smooth textures"
    } else if (temperature < 273) {
      baseDescription = "A cold terrestrial planet"
      surfaceFeatures = `${temperatureEffect}, large polar ice caps, frozen regions dominating the surface, some exposed rocky areas`
      atmosphere = "thin atmosphere with ice crystal clouds"
      textureStyle = "flat graphic style with ice-dominated palette, subtle terrain variations"
    } else if (temperature < 373) {
      baseDescription = "An Earth-like terrestrial planet"
      surfaceFeatures = `${temperatureEffect}, blue liquid water oceans, diverse continents, visible cloud patterns, potentially habitable conditions`
      atmosphere = "breathable nitrogen-oxygen atmosphere with water vapor and dynamic clouds"
      textureStyle = "flat graphic style with vivid blues, greens and browns, white clouds"
    } else if (temperature < 600) {
      baseDescription = "A hot terrestrial planet"
      surfaceFeatures = `${temperatureEffect}, dry rocky surface, desert-like terrain with sand dunes, no surface water, deep impact craters`
      atmosphere = "hot dry atmosphere with dust storms"
      textureStyle = "flat graphic style with red-brown desert tones, minimal features"
    } else if (temperature < 1000) {
      baseDescription = "A very hot terrestrial planet"
      surfaceFeatures = `${temperatureEffect}, scorched surface with heat cracks, extreme volcanic activity, glowing hot spots`
      atmosphere = "thick toxic atmosphere with sulfur compounds and extreme greenhouse effect"
      textureStyle = "flat graphic style with orange-red heat signature, dark volcanic textures"
    } else {
      baseDescription = "An extremely hot lava world"
      surfaceFeatures = `${temperatureEffect}, molten surface with rivers of lava, constant volcanic eruptions, glowing magma oceans`
      atmosphere = "ultra-thick toxic atmosphere with vaporized rock and extreme heat distortion"
      textureStyle = "flat graphic style with bright glowing lava, deep red-orange color scheme, dramatic contrast"
    }
  }

  // Adjust lighting based on star type
  if (starType === "red-dwarf") {
    lighting = "illuminated by dim reddish-orange light from a red dwarf star, creating deep shadows and warm tones"
  } else if (starType === "blue-giant") {
    lighting = "illuminated by intense blue-white light from a blue giant star, creating sharp bright highlights and cool tones"
  } else {
    lighting = "illuminated by bright yellow-white light from a sun-like star, balanced natural lighting"
  }

  // Gravity effects on features
  const gravityDesc = gravity > 2 
    ? "with fine-grained compressed surface features due to high gravity" 
    : gravity < 0.5 
    ? "with large smooth features and expanded atmospheric layers due to low gravity"
    : "with moderate-scale surface features"

  // Build the final optimized prompt for flat graphic style
  const fullPrompt = `Flat graphic exoplanet illustration: ${baseDescription}. 
Style: ${textureStyle}, minimal gradients, subtle cell-shading to suggest spherical curvature, no photorealistic noise or glossy reflections.
Surface: ${surfaceFeatures}, ${gravityDesc}.
Atmosphere: ${atmosphere}.
Temperature: ${temperature}K affecting surface characteristics.
Lighting: ${lighting} from upper-left direction, rim-lit edge.
View: full-sphere centered view, clean planet silhouette against black space.
Composition: scientific art-plate style inspired by NASA JPL mission posters, no text overlays, no stars or extra objects in background, balanced composition, limited color palette driven by temperature (${temperature}K) and atmospheric composition.
Technical: high-resolution square 1:1 format, vector-like crisp details, accurate representation of planetary physics, educational and inspiring.`

  return fullPrompt
}
