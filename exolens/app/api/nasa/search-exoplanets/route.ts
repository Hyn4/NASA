import { NextResponse } from "next/server"

const NASA_API_BASE = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url)
  const query = searchParams.get("q") || ""
  const type = searchParams.get("type") || "star" // 'star' or 'planet'

  if (!query) {
    return NextResponse.json(
      { error: "Search query is required" },
      { status: 400 }
    )
  }

  try {
    let tapQuery = ""
    
    if (type === "star") {
      // Buscar por nome da estrela
      tapQuery = `
        SELECT 
          pl_name,
          hostname,
          discoverymethod,
          disc_year,
          disc_facility,
          pl_rade,
          pl_masse,
          pl_orbper,
          pl_orbsmax,
          pl_eqt,
          pl_insol,
          pl_dens,
          st_spectype,
          st_teff,
          st_rad,
          st_mass,
          st_logg,
          sy_dist,
          sy_vmag,
          default_flag
        FROM ps 
        WHERE hostname LIKE '%${query}%'
        ORDER BY hostname, pl_orbper ASC
      `
    } else {
      // Buscar por nome do planeta
      tapQuery = `
        SELECT 
          pl_name,
          hostname,
          discoverymethod,
          disc_year,
          disc_facility,
          pl_rade,
          pl_masse,
          pl_orbper,
          pl_orbsmax,
          pl_eqt,
          pl_insol,
          pl_dens,
          st_spectype,
          st_teff,
          st_rad,
          st_mass,
          st_logg,
          sy_dist,
          sy_vmag,
          default_flag
        FROM ps 
        WHERE pl_name LIKE '%${query}%'
        ORDER BY hostname, pl_orbper ASC
      `
    }

    const params = new URLSearchParams({
      query: tapQuery,
      format: "json",
    })

    const response = await fetch(`${NASA_API_BASE}?${params}`, {
      headers: {
        "User-Agent": "ExoVerse-3D-Explorer/1.0",
      },
      next: { revalidate: 3600 } // Cache por 1 hora
    })

    if (!response.ok) {
      throw new Error(`NASA API error: ${response.statusText}`)
    }

    const data = await response.json()

    if (!data || data.length === 0) {
      return NextResponse.json({
        results: [],
        count: 0,
        message: "No exoplanets found matching your search"
      })
    }

    // Transformar e agrupar por estrela
    const groupedByHost: { [key: string]: any[] } = {}
    
    data.forEach((planet: any) => {
      const host = planet.hostname || "Unknown"
      if (!groupedByHost[host]) {
        groupedByHost[host] = []
      }
      groupedByHost[host].push({
        pl_name: planet.pl_name,
        hostname: planet.hostname,
        discoverymethod: planet.discoverymethod,
        disc_year: planet.disc_year,
        disc_facility: planet.disc_facility,
        pl_rade: planet.pl_rade,
        pl_masse: planet.pl_masse,
        pl_orbper: planet.pl_orbper,
        pl_orbsmax: planet.pl_orbsmax,
        pl_eqt: planet.pl_eqt,
        pl_insol: planet.pl_insol,
        pl_dens: planet.pl_dens,
        st_spectype: planet.st_spectype,
        st_teff: planet.st_teff,
        st_rad: planet.st_rad,
        st_mass: planet.st_mass,
        st_logg: planet.st_logg,
        sy_dist: planet.sy_dist,
        sy_vmag: planet.sy_vmag,
        default_flag: planet.default_flag
      })
    })

    return NextResponse.json({
      results: groupedByHost,
      count: data.length,
      stars: Object.keys(groupedByHost).length,
      query: query,
      type: type
    })

  } catch (error) {
    console.error("Error fetching NASA data:", error)
    return NextResponse.json(
      {
        error: "Failed to fetch exoplanet data",
        message: error instanceof Error ? error.message : "Unknown error",
      },
      { status: 500 }
    )
  }
}
