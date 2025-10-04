"use client"

import { useEffect, useRef } from "react"

export function MicrolensingAnimation() {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext("2d")
    if (!ctx) return

    canvas.width = 400
    canvas.height = 300

    let lensX = -50
    const backgroundStarX = 200
    const backgroundStarY = 100
    const lensY = 150

    // Store light curve points
    let lightCurve: { x: number; y: number }[] = []

    const animate = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height)

      // Draw background star
      ctx.fillStyle = "#ffeb3b"
      ctx.beginPath()
      ctx.arc(backgroundStarX, backgroundStarY, 8, 0, Math.PI * 2)
      ctx.fill()

      // --- Lensing Calculation ---
      // Main lensing from the star, using a Gaussian for a smooth curve
      const distance = Math.abs(lensX - backgroundStarX)
      const sigma = 40 // Width of the lensing effect
      const lensingStrength = 1.5 * Math.exp(-(distance ** 2) / (2 * sigma ** 2)) // Reduced peak

      // Planet orbiting the lens star
      const planetAngle = lensX * 0.05
      const planetOrbitRadius = 25
      const planetX = lensX + Math.cos(planetAngle) * planetOrbitRadius
      const planetY = lensY + Math.sin(planetAngle) * planetOrbitRadius

      // Planetary lensing spike, also a smooth Gaussian curve
      const planetDistance = Math.abs(planetX - backgroundStarX)
      const spikeSigma = 8 // Wider spike for smoothness
      const planetSpike = 0.3 * Math.exp(-(planetDistance ** 2) / (2 * spikeSigma ** 2)) // Reduced peak

      const totalMagnification = 1 + lensingStrength + planetSpike
      const lensedEffect = lensingStrength + planetSpike

      // Draw lensed light (magnified background star)
      ctx.fillStyle = `rgba(255, 235, 59, ${Math.min(1, 0.5 + lensedEffect / 3)})`
      ctx.beginPath()
      ctx.arc(backgroundStarX, backgroundStarY, 8 + lensedEffect * 5, 0, Math.PI * 2)
      ctx.fill()

      // Draw foreground star (lens)
      ctx.fillStyle = "#ff6b4a"
      ctx.beginPath()
      ctx.arc(lensX, lensY, 12, 0, Math.PI * 2)
      ctx.fill()

      // Draw planet
      ctx.fillStyle = "#4682b4"
      ctx.beginPath()
      ctx.arc(planetX, planetY, 6, 0, Math.PI * 2)
      ctx.fill()

      // --- Light Curve Graph ---
      // Baseline
      ctx.strokeStyle = "#666"
      ctx.lineWidth = 1
      ctx.beginPath()
      ctx.moveTo(50, 250)
      ctx.lineTo(350, 250)
      ctx.stroke()

      // Update light curve data
      const graphX = ((lensX + 50) / 500) * 300 + 50
      const curveY = 250 - (totalMagnification - 1) * 40

      if (lensX >= -50) {
        if (!lightCurve.length || lightCurve[lightCurve.length - 1].x < graphX) {
          lightCurve.push({ x: graphX, y: curveY })
        }
      }

      // Draw magnification curve from stored points
      ctx.strokeStyle = "#4169e1"
      ctx.lineWidth = 2
      ctx.beginPath()
      lightCurve.forEach((p, i) => {
        if (i === 0) ctx.moveTo(p.x, p.y)
        else ctx.lineTo(p.x, p.y)
      })
      ctx.stroke()

      // Labels
      ctx.fillStyle = "#999"
      ctx.font = "12px sans-serif"
      ctx.fillText("Magnification", 10, 230)
      ctx.fillText("Time →", 320, 270)

      // Move lens star at a slower speed
      lensX += 1
      if (lensX > 450) {
        lensX = -50
        lightCurve = [] // Reset curve for next pass
      }

      requestAnimationFrame(animate)
    }

    const animationId = requestAnimationFrame(animate)

    return () => cancelAnimationFrame(animationId)
  }, [])

  return <canvas ref={canvasRef} className="w-full h-auto max-w-md mx-auto" />
}
