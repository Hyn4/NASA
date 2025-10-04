import { NextResponse } from "next/server"

// NASA Exoplanet Archive API endpoint
const NASA_API_BASE = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url)
  const systemName = searchParams.get("system") || "TRAPPIST-1"

  try {
    // Query the NASA Exoplanet Archive using TAP (Table Access Protocol)
    const query = `
      SELECT 
        pl_name, 
        pl_rade, 
        pl_masse, 
        pl_orbper, 
        pl_orbsmax, 
        pl_eqt, 
        disc_year,
        hostname,
        st_rad,
        st_teff
      FROM ps 
      WHERE hostname LIKE '${systemName}%'
      ORDER BY pl_orbper ASC
    `

    const params = new URLSearchParams({
      query: query,
      format: "json",
    })

    const response = await fetch(`${NASA_API_BASE}?${params}`, {
      headers: {
        "User-Agent": "ExoVerse-3D-Explorer/1.0",
      },
    })

    if (!response.ok) {
      throw new Error(`NASA API error: ${response.statusText}`)
    }

    const data = await response.json()

    // Transform the data into a more usable format
    const planets = data.map((planet: any) => ({
      name: planet.pl_name,
      radius: planet.pl_rade || 1.0, // Earth radii
      mass: planet.pl_masse || 1.0, // Earth masses
      period: planet.pl_orbper || 1.0, // days
      distance: planet.pl_orbsmax || 0.1, // AU
      temperature: planet.pl_eqt || 288, // Kelvin
      discovered: planet.disc_year || "Unknown",
      hostStar: planet.hostname,
      starRadius: planet.st_rad || 1.0,
      starTemp: planet.st_teff || 5778,
    }))

    return NextResponse.json({
      system: systemName,
      planets: planets,
      count: planets.length,
    })
  } catch (error) {
    console.error("Error fetching NASA data:", error)
    return NextResponse.json(
      {
        error: "Failed to fetch exoplanet data",
        message: error instanceof Error ? error.message : "Unknown error",
      },
      { status: 500 },
    )
  }
}
