"use client"

import { useEffect, useRef } from "react"

export function DirectImagingAnimation() {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext("2d")
    if (!ctx) return

    canvas.width = 400
    canvas.height = 300

    let coronagraphActive = false
    let time = 0

    const animate = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height)

      const centerX = 200
      const centerY = 120
      const starRadius = 40
      const planetDistance = 80
      const planetAngle = time * 0.02

      // Toggle coronagraph every 3 seconds
      if (Math.floor(time / 180) % 2 === 1) {
        coronagraphActive = true
      } else {
        coronagraphActive = false
      }

      if (!coronagraphActive) {
        // Draw bright star overwhelming everything
        const gradient = ctx.createRadialGradient(centerX, centerY, 0, centerX, centerY, starRadius * 2)
        gradient.addColorStop(0, "rgba(255, 235, 59, 1)")
        gradient.addColorStop(0.3, "rgba(255, 235, 59, 0.8)")
        gradient.addColorStop(0.6, "rgba(255, 235, 59, 0.4)")
        gradient.addColorStop(1, "rgba(255, 235, 59, 0)")
        ctx.fillStyle = gradient
        ctx.beginPath()
        ctx.arc(centerX, centerY, starRadius * 2, 0, Math.PI * 2)
        ctx.fill()

        // Star core
        ctx.fillStyle = "#ffeb3b"
        ctx.beginPath()
        ctx.arc(centerX, centerY, starRadius, 0, Math.PI * 2)
        ctx.fill()

        // Label
        ctx.fillStyle = "#999"
        ctx.font = "14px sans-serif"
        ctx.fillText("Without Coronagraph", 120, 250)
        ctx.fillText("(Planet hidden by starlight)", 100, 270)
      } else {
        // Draw coronagraph blocking star
        ctx.fillStyle = "#333"
        ctx.beginPath()
        ctx.arc(centerX, centerY, starRadius + 5, 0, Math.PI * 2)
        ctx.fill()

        // Draw faint star glow
        ctx.fillStyle = "rgba(255, 235, 59, 0.2)"
        ctx.beginPath()
        ctx.arc(centerX, centerY, starRadius + 10, 0, Math.PI * 2)
        ctx.fill()

        // Draw planet now visible
        const planetX = centerX + Math.cos(planetAngle) * planetDistance
        const planetY = centerY + Math.sin(planetAngle) * planetDistance

        // Planet glow
        ctx.fillStyle = "rgba(70, 130, 180, 0.4)"
        ctx.beginPath()
        ctx.arc(planetX, planetY, 12, 0, Math.PI * 2)
        ctx.fill()

        // Planet
        ctx.fillStyle = "#4682b4"
        ctx.beginPath()
        ctx.arc(planetX, planetY, 8, 0, Math.PI * 2)
        ctx.fill()

        // Draw orbit path
        ctx.strokeStyle = "rgba(255, 255, 255, 0.2)"
        ctx.lineWidth = 1
        ctx.setLineDash([5, 5])
        ctx.beginPath()
        ctx.arc(centerX, centerY, planetDistance, 0, Math.PI * 2)
        ctx.stroke()
        ctx.setLineDash([])

        // Label
        ctx.fillStyle = "#4169e1"
        ctx.font = "14px sans-serif"
        ctx.fillText("With Coronagraph", 130, 250)
        ctx.fillText("(Planet visible!)", 150, 270)
      }

      time++
      requestAnimationFrame(animate)
    }

    const animationId = requestAnimationFrame(animate)

    return () => cancelAnimationFrame(animationId)
  }, [])

  return <canvas ref={canvasRef} className="w-full h-auto max-w-md mx-auto" />
}
