"use client"

import { useEffect, useRef } from "react"
import type { PlanetParameters } from "./planet-creator"

interface LightCurveSimulationProps {
  parameters: PlanetParameters
}

export function LightCurveSimulation({ parameters }: LightCurveSimulationProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext("2d")
    if (!ctx) return

    const width = canvas.width
    const height = canvas.height

    ctx.clearRect(0, 0, width, height)

    // Calculate transit depth based on planet and star radii
    const starRadius = 1.0 // Solar radii
    const planetRadius = parameters.radius * 0.00916 // Convert Earth radii to Solar radii
    const transitDepth = Math.pow(planetRadius / starRadius, 2) * 100 // Percentage

    // Draw axes
    ctx.strokeStyle = "#666"
    ctx.lineWidth = 2
    ctx.beginPath()
    ctx.moveTo(50, 50)
    ctx.lineTo(50, height - 50)
    ctx.lineTo(width - 50, height - 50)
    ctx.stroke()

    // Draw labels
    ctx.fillStyle = "#999"
    ctx.font = "14px sans-serif"
    ctx.fillText("Brightness", 10, 30)
    ctx.fillText("Time", width - 80, height - 20)
    ctx.fillText("100%", 10, height - 50)
    ctx.fillText(`${(100 - transitDepth).toFixed(2)}%`, 10, height - 50 + transitDepth * 3)

    const numPoints = 200
    const dataPoints: { x: number; y: number }[] = []
    const baselineY = height - 50
    const transitWidth = 100
    const transitStart = width / 2 - transitWidth / 2
    const transitEnd = width / 2 + transitWidth / 2

    for (let i = 0; i < numPoints; i++) {
      const x = 50 + (i / numPoints) * (width - 100)
      let y = baselineY

      // Add transit dip
      if (x >= transitStart && x <= transitEnd) {
        const transitProgress = (x - transitStart) / transitWidth

        // Smooth ingress and egress
        if (transitProgress < 0.1) {
          const ingressProgress = transitProgress / 0.1
          y = baselineY + transitDepth * 3 * ingressProgress
        } else if (transitProgress > 0.9) {
          const egressProgress = (1 - transitProgress) / 0.1
          y = baselineY + transitDepth * 3 * egressProgress
        } else {
          y = baselineY + transitDepth * 3
        }
      }

      // Add realistic noise (photon noise)
      const noise = (Math.random() - 0.5) * 2 // ±1 pixel noise
      y += noise

      dataPoints.push({ x, y })
    }

    // Draw data points
    ctx.fillStyle = "#4169e1"
    dataPoints.forEach((point) => {
      ctx.beginPath()
      ctx.arc(point.x, point.y, 2, 0, Math.PI * 2)
      ctx.fill()
    })

    // Draw best-fit line through the data
    ctx.strokeStyle = "#ff6b4a"
    ctx.lineWidth = 2
    ctx.beginPath()
    ctx.moveTo(dataPoints[0].x, dataPoints[0].y)

    // Use moving average for smooth line
    const windowSize = 5
    for (let i = 0; i < dataPoints.length; i++) {
      const start = Math.max(0, i - Math.floor(windowSize / 2))
      const end = Math.min(dataPoints.length, i + Math.ceil(windowSize / 2))
      const avgY = dataPoints.slice(start, end).reduce((sum, p) => sum + p.y, 0) / (end - start)
      ctx.lineTo(dataPoints[i].x, avgY)
    }
    ctx.stroke()

    // Draw transit depth annotation
    ctx.strokeStyle = "#00d9ff"
    ctx.setLineDash([5, 5])
    ctx.beginPath()
    ctx.moveTo(width - 100, baselineY)
    ctx.lineTo(width - 100, baselineY + transitDepth * 3)
    ctx.stroke()
    ctx.setLineDash([])

    ctx.fillStyle = "#00d9ff"
    ctx.font = "12px sans-serif"
    ctx.fillText(`Depth: ${transitDepth.toFixed(3)}%`, width - 150, baselineY + transitDepth * 3 + 20)
  }, [parameters])

  return (
    <div className="w-full">
      <canvas ref={canvasRef} width={600} height={300} className="w-full h-auto border border-border rounded-lg" />
      <div className="mt-4 text-sm text-muted-foreground space-y-2">
        <p>
          <span className="font-semibold text-foreground">Transit Depth:</span>{" "}
          {(Math.pow((parameters.radius * 0.00916) / 1.0, 2) * 100).toFixed(4)}%
        </p>
        <p>
          <span className="font-semibold text-foreground">Detection Difficulty:</span>{" "}
          {parameters.radius < 1.5
            ? "Very Hard"
            : parameters.radius < 3
              ? "Hard"
              : parameters.radius < 6
                ? "Moderate"
                : "Easy"}
        </p>
        <p className="text-xs italic">
          Blue dots represent individual brightness measurements with realistic photon noise. Red line shows the
          best-fit model.
        </p>
      </div>
    </div>
  )
}
