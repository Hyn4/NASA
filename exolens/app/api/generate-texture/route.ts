// app/api/generate-texture/route.ts
import { NextResponse } from "next/server"
import { GoogleAuth } from 'google-auth-library'

export async function POST(request: Request) {
  try {
    const body = await request.json()
    const { radius, temperature, mass, starType, planetType, distance } = body
    console.log(body)

    // Build a detailed prompt for AI image generation
    const prompt = buildTexturePrompt({ radius, temperature, mass, starType, planetType, distance })

    // Carrega as variáveis do .env
    const GOOGLE_API_KEY = process.env.GOOGLE_API_KEY;
    const GOOGLE_SERVICE_ACCOUNT_TYPE = process.env.GOOGLE_SERVICE_ACCOUNT_TYPE;
    const GOOGLE_SERVICE_ACCOUNT_PROJECT_ID = process.env.GOOGLE_SERVICE_ACCOUNT_PROJECT_ID;
    const GOOGLE_SERVICE_ACCOUNT_PRIVATE_KEY_ID = process.env.GOOGLE_SERVICE_ACCOUNT_PRIVATE_KEY_ID;
    const GOOGLE_SERVICE_ACCOUNT_PRIVATE_KEY = process.env.GOOGLE_SERVICE_ACCOUNT_PRIVATE_KEY?.replace(/\\n/g, '\n');
    const GOOGLE_SERVICE_ACCOUNT_CLIENT_EMAIL = process.env.GOOGLE_SERVICE_ACCOUNT_CLIENT_EMAIL;
    const GOOGLE_SERVICE_ACCOUNT_CLIENT_ID = process.env.GOOGLE_SERVICE_ACCOUNT_CLIENT_ID;
    const GOOGLE_SERVICE_ACCOUNT_AUTH_URI = process.env.GOOGLE_SERVICE_ACCOUNT_AUTH_URI;
    const GOOGLE_SERVICE_ACCOUNT_TOKEN_URI = process.env.GOOGLE_SERVICE_ACCOUNT_TOKEN_URI;
    const GOOGLE_SERVICE_ACCOUNT_AUTH_PROVIDER_X509_CERT_URL = process.env.GOOGLE_SERVICE_ACCOUNT_AUTH_PROVIDER_X509_CERT_URL;
    const GOOGLE_SERVICE_ACCOUNT_CLIENT_X509_CERT_URL = process.env.GOOGLE_SERVICE_ACCOUNT_CLIENT_X509_CERT_URL;
    const GOOGLE_SERVICE_ACCOUNT_UNIVERSE_DOMAIN = process.env.GOOGLE_SERVICE_ACCOUNT_UNIVERSE_DOMAIN;

    if (!GOOGLE_API_KEY || !GOOGLE_SERVICE_ACCOUNT_PRIVATE_KEY) {
      console.warn('Google API key or service account not configured, returning placeholder')
      return NextResponse.json({
        success: true,
        prompt: prompt,
        textureUrl: `/placeholder.svg?height=1024&width=1024&query=${encodeURIComponent(prompt)}`,
        message: "Google API key or service account not configured. Using placeholder image.",
      })
    }

    const serviceAccountKey = {
      type: GOOGLE_SERVICE_ACCOUNT_TYPE,
      project_id: GOOGLE_SERVICE_ACCOUNT_PROJECT_ID,
      private_key_id: GOOGLE_SERVICE_ACCOUNT_PRIVATE_KEY_ID,
      private_key: GOOGLE_SERVICE_ACCOUNT_PRIVATE_KEY,
      client_email: GOOGLE_SERVICE_ACCOUNT_CLIENT_EMAIL,
      client_id: GOOGLE_SERVICE_ACCOUNT_CLIENT_ID,
      auth_uri: GOOGLE_SERVICE_ACCOUNT_AUTH_URI,
      token_uri: GOOGLE_SERVICE_ACCOUNT_TOKEN_URI,
      auth_provider_x509_cert_url: GOOGLE_SERVICE_ACCOUNT_AUTH_PROVIDER_X509_CERT_URL,
      client_x509_cert_url: GOOGLE_SERVICE_ACCOUNT_CLIENT_X509_CERT_URL,
      universe_domain: GOOGLE_SERVICE_ACCOUNT_UNIVERSE_DOMAIN,
    };

    const auth = new GoogleAuth({
      credentials: serviceAccountKey,
      scopes: 'https://www.googleapis.com/auth/cloud-platform',
    });
    const accessToken = await auth.getAccessToken();

    if (!accessToken) {
        throw new Error('Failed to acquire an access token.');
    }

    // Call Google Imagen API to generate the texture
    const imagenResponse = await fetch(
      // A URL foi atualizada para o endpoint do Vertex AI
      // Substitua ${LOCATION}, ${PROJECT_ID} e ${MODEL_ID} pelos seus valores
      `https://us-central1-aiplatform.googleapis.com/v1/projects/exolens/locations/us-central1/publishers/google/models/imagen-3.0-generate-002:predict`,
      {
        method: 'POST',
        headers: {
          // A autenticação agora é feita com um Bearer Token
          'Authorization': `Bearer ${accessToken}`,
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

    // Salva o base64 em um arquivo para debug
    const fs = require('fs');
    const debugFilePath = process.cwd() + '/debug_texture_base64.txt';
    fs.writeFileSync(debugFilePath, `data:image/png;base64,${imageBase64}`);

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
  temperature: number // Continua sendo a temperatura da ESTRELA
  mass: number
  starType: string
  planetType: string
  distance: number // NOVO PARÂMETRO: Distância do planeta à estrela em AU
}): string {
  // Renomeia 'temperature' para 'starTemperature' para clareza e adiciona 'distance'
  const { radius, temperature: starTemperature, mass, starType, planetType, distance } = params

  // Função auxiliar para obter o raio relativo da estrela (Sol = 1)
  const getStarRadius = (type: string): number => {
    switch (type) {
      case "red-dwarf": return 0.3;
      case "blue-giant": return 10;
      default: return 1.0; // sun-like
    }
  };

  // Função para calcular uma temperatura de superfície estimada para o planeta
  const calculatePlanetSurfaceTemp = (starTemp: number, starRad: number, dist: number): number => {
    // Evita divisão por zero para planetas muito próximos
    if (dist <= 0) return 10000; 

    const earthEquilibriumTemp = 288; // Temp. média da Terra em Kelvin (~15°C)
    const sunTemp = 5778;

    // Fórmula de temperatura de equilíbrio simplificada: T_planeta ∝ sqrt(R_estrela) * T_estrela / sqrt(D_planeta)
    // Isso modela como a energia recebida diminui com a distância.
    const planetTemp = earthEquilibriumTemp * Math.sqrt(starRad) * (starTemp / sunTemp) / Math.sqrt(dist);
    return planetTemp;
  };
  
  const starRadius = getStarRadius(starType);
  // **A VARIÁVEL-CHAVE:** A temperatura calculada da superfície do planeta
  const planetSurfaceTemp = calculatePlanetSurfaceTemp(starTemperature, starRadius, distance);

  let baseDescription = ""
  let surfaceFeatures = ""
  let atmosphere = ""
  let textureStyle = ""
  let gravity = mass / (radius * radius)

  // --- TODA A LÓGICA ABAIXO AGORA USA 'planetSurfaceTemp' ---

  // Determina o efeito da temperatura na superfície
  let temperatureEffect = ""
  if (planetSurfaceTemp > 1500) {
    temperatureEffect = "molten lava surface with glowing magma cracks and volcanic eruptions"
  } else if (planetSurfaceTemp > 800) {
    temperatureEffect = "cracked basalt surface with volcanic features and dark lava flows"
  } else if (planetSurfaceTemp > 400) {
    temperatureEffect = "dry rocky regolith with impact craters and barren landscape"
  } else if (planetSurfaceTemp > 273) { // Ponto de congelamento da água
    temperatureEffect = "rocky surface with weathering patterns and mineral deposits, potential for liquid water"
  } else if (planetSurfaceTemp > 200) {
    temperatureEffect = "partially frozen surface with ice formations and frozen lakes"
  } else {
    temperatureEffect = "completely frozen icy crust with methane ice and nitrogen frost"
  }

  // Determina as características do planeta com base no tipo e na temperatura da superfície
  if (planetType === "Gas Giant") {
    baseDescription = "gas giant"
    surfaceFeatures = `turbulent atmospheric storms, horizontal cloud bands similar to Jupiter with ${gravity > 2 ? 'fine-grained compressed' : 'large swirling'} storm systems, great red spot-like vortices`
    atmosphere = "thick hydrogen and helium atmosphere with dramatic color variations from ammonia, methane, and phosphorus compounds"
    textureStyle = "seamless tileable texture map with bold horizontal banded patterns, smooth color transitions between bands"
  } else if (planetType === "Neptune-like") {
    baseDescription = "ice giant"
    surfaceFeatures = `methane-rich clouds with subtle horizontal atmospheric bands, ${gravity > 1.5 ? 'compressed' : 'expanded'} storm features, high-altitude cirrus-like formations`
    atmosphere = "thick atmosphere dominated by methane giving deep blue-green coloration with wispy white clouds"
    textureStyle = "seamless tileable texture map with smooth gradients, blue-green dominant palette, subtle cloud streaks"
  } else if (planetType === "Super-Earth") {
    if (planetSurfaceTemp < 273) {
      baseDescription = "large frozen rocky super-Earth"
      surfaceFeatures = `frozen ocean surfaces, massive ice sheets with cracks and pressure ridges, ${gravity > 2 ? 'flat ice plains' : 'jagged ice mountains'}, exposed rocky continents with snow cover`
      atmosphere = "thin atmosphere creating frost patterns and ice crystal formations"
      textureStyle = "seamless tileable texture map with white-blue ice textures, glacial flow patterns, polar caps"
    } else if (planetSurfaceTemp < 373) {
      baseDescription = "large Earth-like super-Earth with liquid water"
      surfaceFeatures = `vast blue oceans with wave patterns, large continents with varied biomes, mountain ranges, river systems, forest regions, desert areas, polar ice caps, ${gravity > 2 ? 'flat plains' : 'mountain ranges'}`
      atmosphere = "white cloud formations, cyclone systems, scattered cumulus clouds"
      textureStyle = "seamless tileable texture map with vivid blue oceans, green-brown continents, white clouds as overlay"
    } else {
      baseDescription = "large hot volcanic super-Earth"
      surfaceFeatures = `${temperatureEffect}, active volcanoes with lava flows, extensive volcanic plains, impact craters, no water bodies`
      atmosphere = "volcanic ash clouds and heat haze patterns"
      textureStyle = "seamless tileable texture map with red-orange volcanic glow, dark basalt textures, dramatic lava rivers"
    }
  } else {
    // Terrestrial
    if (planetSurfaceTemp < 200) {
      baseDescription = "frozen terrestrial world"
      surfaceFeatures = `${temperatureEffect}, completely frozen surface with methane and nitrogen ice, no liquid water, smooth ice plains with minimal features`
      atmosphere = "very thin atmosphere creating minimal frost patterns"
      textureStyle = "seamless tileable texture map with pale blue-white ice, smooth uniform textures, subtle albedo variations"
    } else if (planetSurfaceTemp < 273) {
      baseDescription = "cold terrestrial world"
      surfaceFeatures = `${temperatureEffect}, large polar ice caps extending toward equator, frozen regions, exposed rocky terrain, glacial formations`
      atmosphere = "thin atmosphere with ice crystal cloud patterns"
      textureStyle = "seamless tileable texture map with ice-dominated palette, polar regions clearly visible, rocky patches"
    } else if (planetSurfaceTemp < 373) {
      baseDescription = "Earth-like terrestrial world with liquid water"
      surfaceFeatures = `${temperatureEffect}, blue liquid water oceans, diverse continents with varied terrain, mountain ranges, valleys, plains, river deltas, polar ice caps`
      atmosphere = "scattered white clouds, storm systems, dynamic weather patterns"
      textureStyle = "seamless tileable texture map with vivid blue oceans (60-70% coverage), green and brown landmasses, white polar caps, cloud layer"
    } else if (planetSurfaceTemp < 600) {
      baseDescription = "hot, dry terrestrial world"
      surfaceFeatures = `${temperatureEffect}, extensive desert terrain with dune fields, no surface water, deep impact craters, weathered highlands`
      atmosphere = "hot dry atmosphere with occasional dust storm patterns"
      textureStyle = "seamless tileable texture map with red-brown desert tones, orange sand, dark rocky regions, minimal contrast"
    } else if (planetSurfaceTemp < 1000) {
      baseDescription = "very hot volcanic terrestrial world"
      surfaceFeatures = `${temperatureEffect}, scorched surface with thermal stress cracks, extreme volcanic activity, glowing hot spots, lava lakes`
      atmosphere = "thick toxic atmosphere with sulfur-tinted clouds"
      textureStyle = "seamless tileable texture map with orange-red heat signature, dark volcanic plains, glowing lava features"
    } else {
      baseDescription = "extremely hot lava world"
      surfaceFeatures = `${temperatureEffect}, molten surface with lava rivers, constant volcanic eruptions, glowing magma oceans, semi-solid crust plates floating on magma`
      atmosphere = "ultra-thick toxic atmosphere with vaporized rock creating orange-red haze"
      textureStyle = "seamless tileable texture map with bright glowing lava, deep red-orange color scheme, high contrast between molten and cooling areas"
    }
  }

  // A iluminação ainda depende do tipo da estrela, então isso permanece igual.
  let lightingDesc = ""
  if (starType === "red-dwarf") {
    lightingDesc = "illuminated by dim reddish-orange light from a red dwarf star, warm color temperature, deep shadows in craters"
  } else if (starType === "blue-giant") {
    lightingDesc = "illuminated by intense blue-white light from a blue giant star, cool color temperature, sharp bright highlights"
  } else {
    lightingDesc = "illuminated by bright yellow-white light from a sun-like star, neutral color temperature, balanced lighting"
  }

  // Efeitos da gravidade permanecem iguais.
  const gravityDesc = gravity > 2 
    ? "with fine-grained compressed surface features due to high gravity, flat terrain dominates" 
    : gravity < 0.5 
    ? "with large smooth features and tall formations due to low gravity"
    : "with moderate-scale surface features and varied topography"

  // Prompt final atualizado com a temperatura correta
const fullPrompt = `Realistic flat texture map of a ${baseDescription}, designed for 3D sphere wrapping.

FORMAT: Seamless, tileable, SQUARE 2D texture map (1:1 aspect ratio). The texture MUST be designed so the left edge connects perfectly to the right edge to allow for horizontal repeating.

STYLE: ${textureStyle}, photorealistic surface rendering with accurate color grading. Natural lighting with soft shadows. High detail level showing microscopic surface variations.

SURFACE FEATURES: ${surfaceFeatures}, ${gravityDesc}. Features should be distributed evenly across the square map.

ATMOSPHERE & WEATHER: ${atmosphere}. ${planetType !== "Gas Giant" && planetType !== "Neptune-like" ? "Clouds should be rendered as a semi-transparent layer if present." : ""}

COLOR & LIGHTING: ${lightingDesc}. Planet's estimated surface temperature is ${planetSurfaceTemp.toFixed(0)}K (from a ${starTemperature}K star at ${distance} AU), which directly influences the color palette and surface state.

TECHNICAL SPECIFICATIONS:
- Seamless horizontal wrap (left edge connects to right edge)
- No borders, frames, text overlays, or UI elements
- No planet sphere visible - this is a flat unwrapped surface map
- High resolution suitable for close-up 3D rendering
- Clean, scientifically accurate representation

OUTPUT: Pure square texture map only, ready for 3D texture application.
-
`

  return fullPrompt
}