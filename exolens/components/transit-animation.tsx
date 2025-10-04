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

    const animate = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height)

      const distanceFromCenter = Math.abs(planetX - starX)
      const isTransiting = distanceFromCenter < starRadius
      const transitProgress = isTransiting ? 1 - distanceFromCenter / starRadius : 0
      const brightness = 1 - transitProgress * 0.15 // Reduce brightness by up to 15%

      // Draw star with dynamic brightness
      const gradient = ctx.createRadialGradient(starX, starY, 0, starX, starY, starRadius)
      gradient.addColorStop(0, `rgba(255, 235, 59, ${brightness})`)
      gradient.addColorStop(0.5, `rgba(255, 167, 38, ${brightness})`)
      gradient.addColorStop(1, `rgba(255, 111, 0, ${brightness})`)
      ctx.fillStyle = gradient
      ctx.beginPath()
      ctx.arc(starX, starY, starRadius, 0, Math.PI * 2)
      ctx.fill()

      // Draw star glow with dynamic brightness
      ctx.fillStyle = `rgba(255, 235, 59, ${0.2 * brightness})`
      ctx.beginPath()
      ctx.arc(starX, starY, starRadius + 10, 0, Math.PI * 2)
      ctx.fill()

      // Draw planet
      ctx.fillStyle = "#4682b4"
      ctx.beginPath()
      ctx.arc(planetX, starY, planetRadius, 0, Math.PI * 2)
      ctx.fill()

      // Draw planet shadow on star during transit
      if (isTransiting) {
        ctx.fillStyle = "rgba(0, 0, 0, 0.4)"
        ctx.beginPath()
        ctx.arc(planetX, starY, planetRadius, 0, Math.PI * 2)
        ctx.fill()
      }

      // Draw light curve below
      ctx.strokeStyle = "#666"
      ctx.lineWidth = 1
      ctx.beginPath()
      ctx.moveTo(50, 250)
      ctx.lineTo(350, 250)
      ctx.stroke()

      const curveY = 250 - brightness * 30

      ctx.strokeStyle = "#4169e1"
      ctx.lineWidth = 2
      ctx.beginPath()
      ctx.moveTo(50, 250 - 30)
      ctx.lineTo(planetX * 0.75 + 50, curveY)
      ctx.stroke()

      // Labels
      ctx.fillStyle = "#999"
      ctx.font = "12px sans-serif"
      ctx.fillText("Brightness", 10, 230)
      ctx.fillText("Time →", 320, 270)

      // Move planet
      planetX += 2
      if (planetX > 450) {
        planetX = -50
      }

      requestAnimationFrame(animate)
    }

    const animationId = requestAnimationFrame(animate)

    return () => cancelAnimationFrame(animationId)
  }, [])

  return <canvas ref={canvasRef} className="w-full h-auto max-w-md mx-auto" />
}
