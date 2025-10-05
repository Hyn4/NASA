import { NextResponse } from "next/server"

const NASA_API_BASE = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"

// Sistemas populares para sugestões
const POPULAR_SYSTEMS = [
  "TRAPPIST-1",
  "Kepler-186",
  "Proxima Cen",
  "TOI-700",
  "K2-18",
  "HD 40307",
  "Gliese 667 C",
  "Kepler-452"
]

export async function GET() {
  try {
    const systemsQuery = POPULAR_SYSTEMS.map(s => `'${s}'`).join(',')
    
    const tapQuery = `
      SELECT 
        hostname,
        COUNT(*) as planet_count,
        st_spectype,
        st_teff,
        sy_dist
      FROM ps 
      WHERE hostname IN (${systemsQuery})
      GROUP BY hostname, st_spectype, st_teff, sy_dist
      ORDER BY planet_count DESC
    `

    const params = new URLSearchParams({
      query: tapQuery,
      format: "json",
    })

    const response = await fetch(`${NASA_API_BASE}?${params}`, {
      headers: {
        "User-Agent": "ExoVerse-3D-Explorer/1.0",
      },
      next: { revalidate: 86400 } // Cache por 24 horas
    })

    if (!response.ok) {
      throw new Error(`NASA API error: ${response.statusText}`)
    }

    const data = await response.json()

    return NextResponse.json({
      systems: data,
      count: data.length
    })

  } catch (error) {
    console.error("Error fetching popular systems:", error)
    return NextResponse.json(
      {
        error: "Failed to fetch popular systems",
        message: error instanceof Error ? error.message : "Unknown error",
      },
      { status: 500 }
    )
  }
}
