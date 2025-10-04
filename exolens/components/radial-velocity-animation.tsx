"use client"

import { useEffect, useRef } from "react"

export function RadialVelocityAnimation() {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext("2d")
    if (!ctx) return

    canvas.width = 400
    canvas.height = 300

    let angle = 0
    const centerX = 200
    const centerY = 120
    const starOrbitRadius = 30
    const planetOrbitRadius = 80

    const animate = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height)

      // Draw center of mass
      ctx.fillStyle = "#666"
      ctx.beginPath()
      ctx.arc(centerX, centerY, 3, 0, Math.PI * 2)
      ctx.fill()

      // Draw orbits
      ctx.strokeStyle = "#444"
      ctx.lineWidth = 1
      ctx.setLineDash([5, 5])
      ctx.beginPath()
      ctx.arc(centerX, centerY, starOrbitRadius, 0, Math.PI * 2)
      ctx.stroke()
      ctx.beginPath()
      ctx.arc(centerX, centerY, planetOrbitRadius, 0, Math.PI * 2)
      ctx.stroke()
      ctx.setLineDash([])

      // Calculate positions
      const starX = centerX + Math.cos(angle) * starOrbitRadius
      const starY = centerY + Math.sin(angle) * starOrbitRadius
      const planetX = centerX + Math.cos(angle + Math.PI) * planetOrbitRadius
      const planetY = centerY + Math.sin(angle + Math.PI) * planetOrbitRadius

      // Draw star
      const gradient = ctx.createRadialGradient(starX, starY, 0, starX, starY, 25)
      gradient.addColorStop(0, "#ffeb3b")
      gradient.addColorStop(1, "#ff6f00")
      ctx.fillStyle = gradient
      ctx.beginPath()
      ctx.arc(starX, starY, 25, 0, Math.PI * 2)
      ctx.fill()

      // Draw planet
      ctx.fillStyle = "#4682b4"
      ctx.beginPath()
      ctx.arc(planetX, planetY, 12, 0, Math.PI * 2)
      ctx.fill()

      // Draw velocity graph
      ctx.strokeStyle = "#666"
      ctx.lineWidth = 1
      ctx.beginPath()
      ctx.moveTo(50, 230)
      ctx.lineTo(350, 230)
      ctx.stroke()

      // Draw velocity curve
      const velocity = Math.cos(angle) * 20
      ctx.strokeStyle = "#4169e1"
      ctx.lineWidth = 2
      ctx.beginPath()
      for (let i = 0; i < 300; i++) {
        const x = 50 + i
        const a = ((angle - (300 - i) * 0.02) % (Math.PI * 2)) + Math.PI * 2
        const v = Math.cos(a) * 20
        const y = 230 - v
        if (i === 0) {
          ctx.moveTo(x, y)
        } else {
          ctx.lineTo(x, y)
        }
      }
      ctx.stroke()

      // Labels
      ctx.fillStyle = "#999"
      ctx.font = "12px sans-serif"
      ctx.fillText("Velocity", 10, 220)
      ctx.fillText("Time →", 320, 250)
      ctx.fillText("Wobble", centerX - 20, centerY - 50)

      angle += 0.02

      requestAnimationFrame(animate)
    }

    const animationId = requestAnimationFrame(animate)

    return () => cancelAnimationFrame(animationId)
  }, [])

  return <canvas ref={canvasRef} className="w-full h-auto max-w-md mx-auto" />
}
