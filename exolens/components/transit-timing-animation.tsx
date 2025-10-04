"use client"

import { useEffect, useRef } from "react"

export function TransitTimingAnimation() {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext("2d")
    if (!ctx) return

    canvas.width = 400
    canvas.height = 300

    let time = 0
    const transitPeriod = 120 // frames between transits
    const timingVariation = 10 // frames of variation

    const animate = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height)

      // Draw timeline
      ctx.strokeStyle = "#666"
      ctx.lineWidth = 2
      ctx.beginPath()
      ctx.moveTo(50, 150)
      ctx.lineTo(350, 150)
      ctx.stroke()

      // Calculate expected vs actual transit times
      const numTransits = 3
      for (let i = 0; i < numTransits; i++) {
        const expectedTime = 50 + i * 100
        const actualTime = expectedTime + Math.sin(i * 1.5) * timingVariation

        // Draw expected transit (gray)
        ctx.fillStyle = "#666"
        ctx.beginPath()
        ctx.arc(expectedTime, 150, 4, 0, Math.PI * 2)
        ctx.fill()

        // Draw expected line
        ctx.strokeStyle = "#666"
        ctx.setLineDash([3, 3])
        ctx.beginPath()
        ctx.moveTo(expectedTime, 130)
        ctx.lineTo(expectedTime, 170)
        ctx.stroke()
        ctx.setLineDash([])

        // Draw actual transit (blue)
        if (time > i * transitPeriod) {
          ctx.fillStyle = "#4169e1"
          ctx.beginPath()
          ctx.arc(actualTime, 150, 6, 0, Math.PI * 2)
          ctx.fill()

          // Draw actual line
          ctx.strokeStyle = "#4169e1"
          ctx.lineWidth = 2
          ctx.beginPath()
          ctx.moveTo(actualTime, 130)
          ctx.lineTo(actualTime, 170)
          ctx.stroke()

          // Draw variation arrow
          if (Math.abs(actualTime - expectedTime) > 2) {
            ctx.strokeStyle = "#ff6b4a"
            ctx.lineWidth = 2
            ctx.beginPath()
            ctx.moveTo(expectedTime, 180)
            ctx.lineTo(actualTime, 180)
            ctx.stroke()

            // Arrow head
            const direction = actualTime > expectedTime ? 1 : -1
            ctx.beginPath()
            ctx.moveTo(actualTime, 180)
            ctx.lineTo(actualTime - direction * 5, 175)
            ctx.moveTo(actualTime, 180)
            ctx.lineTo(actualTime - direction * 5, 185)
            ctx.stroke()
          }
        }
      }

      // Draw star and planets at top
      const starX = 200
      const starY = 50

      // Star
      ctx.fillStyle = "#ffeb3b"
      ctx.beginPath()
      ctx.arc(starX, starY, 15, 0, Math.PI * 2)
      ctx.fill()

      // Planet 1 (transiting)
      const planet1Angle = (time * 0.03) % (Math.PI * 2)
      const planet1X = starX + Math.cos(planet1Angle) * 30
      const planet1Y = starY + Math.sin(planet1Angle) * 30
      ctx.fillStyle = "#4169e1"
      ctx.beginPath()
      ctx.arc(planet1X, planet1Y, 6, 0, Math.PI * 2)
      ctx.fill()

      // Planet 2 (perturbing)
      const planet2Angle = (time * 0.02) % (Math.PI * 2)
      const planet2X = starX + Math.cos(planet2Angle) * 50
      const planet2Y = starY + Math.sin(planet2Angle) * 50
      ctx.fillStyle = "#ff6b4a"
      ctx.beginPath()
      ctx.arc(planet2X, planet2Y, 5, 0, Math.PI * 2)
      ctx.fill()

      // Labels
      ctx.fillStyle = "#999"
      ctx.font = "12px sans-serif"
      ctx.fillText("Expected", 10, 125)
      ctx.fillText("Actual", 10, 145)
      ctx.fillText("Timing Variation →", 250, 200)

      time++
      if (time > 360) time = 0

      requestAnimationFrame(animate)
    }

    const animationId = requestAnimationFrame(animate)

    return () => cancelAnimationFrame(animationId)
  }, [])

  return <canvas ref={canvasRef} className="w-full h-auto max-w-md mx-auto" />
}
