"use client"

import { useEffect, useRef } from "react"

export function TransitAnimation() {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext("2d")
    if (!ctx) return

    canvas.width = 400
    canvas.height = 300

    let planetX = -50
    const starX = 200
    const starY = 150
    const starRadius = 60
    const planetRadius = 15

    // Store light curve points
    let lightCurve: { x: number; y: number }[] = []

    const animate = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height)

      // ----- Brightness Calculation -----
      const distanceFromCenter = Math.abs(planetX - starX)
      const isTransiting = distanceFromCenter < starRadius

      // Use a curved easing function for a more realistic light curve
      const normalizedDistance = Math.min(1, distanceFromCenter / starRadius)
      // Create a semi-circular curve for the dip
      const easeInProgress = Math.sqrt(1 - normalizedDistance ** 2)
      const transitProgress = isTransiting ? easeInProgress : 0
      const brightness = 1 - transitProgress * 0.9

      // ----- Star Drawing -----
      const gradient = ctx.createRadialGradient(starX, starY, 0, starX, starY, starRadius)
      gradient.addColorStop(0, `rgba(255, 235, 59, ${brightness})`)
      gradient.addColorStop(0.5, `rgba(255, 167, 38, ${brightness})`)
      gradient.addColorStop(1, `rgba(255, 111, 0, ${brightness})`)
      ctx.fillStyle = gradient
      ctx.beginPath()
      ctx.arc(starX, starY, starRadius, 0, Math.PI * 2)
      ctx.fill()

      // Star Glow
      ctx.fillStyle = `rgba(255, 235, 59, ${0.2 * brightness})`
      ctx.beginPath()
      ctx.arc(starX, starY, starRadius + 10, 0, Math.PI * 2)
      ctx.fill()

      // ----- Planet Drawing -----
      ctx.fillStyle = "#4682b4"
      ctx.beginPath()
      ctx.arc(planetX, starY, planetRadius, 0, Math.PI * 2)
      ctx.fill()

      // Planet's shadow on the star
      if (isTransiting) {
        ctx.fillStyle = "rgba(0, 0, 0, 0.4)"
        ctx.beginPath()
        ctx.arc(planetX, starY, planetRadius, 0, Math.PI * 2)
        ctx.fill()
      }

      // ----- Light Curve Graph -----
      // Axis
      ctx.strokeStyle = "#666"
      ctx.lineWidth = 1
      ctx.beginPath()
      ctx.moveTo(50, 250)
      ctx.lineTo(350, 250)
      ctx.stroke()

      // Add new point to graph
      const graphX = (planetX + 50) * 0.8 // horizontal scale
      const graphY = 250 - brightness * 40 // vertical scale
      if (graphX >= 50 && graphX <= 350) {
        lightCurve.push({ x: graphX, y: graphY })
      }

      // Keep a maximum number of points
      if (lightCurve.length > 300) {
        lightCurve.shift()
      }

      // Draw curve
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
      ctx.fillText("Brightness", 10, 230)
      ctx.fillText("Time →", 320, 270)

      // ----- Planet Movement -----
      planetX += 2
      if (planetX > 450) {
        planetX = -50
        lightCurve = [] // reset curve on restart
      }

      requestAnimationFrame(animate)
    }

    const animationId = requestAnimationFrame(animate)
    return () => cancelAnimationFrame(animationId)
  }, [])

  return <canvas ref={canvasRef} className="w-full h-auto max-w-md mx-auto" />
}