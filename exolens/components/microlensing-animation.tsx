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

    const animate = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height)

      // Draw background star
      ctx.fillStyle = "#ffeb3b"
      ctx.beginPath()
      ctx.arc(backgroundStarX, backgroundStarY, 8, 0, Math.PI * 2)
      ctx.fill()

      // Calculate lensing effect
      const distance = Math.abs(lensX - backgroundStarX)
      const maxLensing = 100
      const lensingStrength = distance < maxLensing ? (1 - distance / maxLensing) * 3 : 0

      // Draw lensed light (magnified background star)
      if (lensingStrength > 0) {
        ctx.fillStyle = `rgba(255, 235, 59, ${0.3 * lensingStrength})`
        ctx.beginPath()
        ctx.arc(backgroundStarX, backgroundStarY, 8 + lensingStrength * 10, 0, Math.PI * 2)
        ctx.fill()
      }

      // Draw foreground star (lens)
      ctx.fillStyle = "#ff6b4a"
      ctx.beginPath()
      ctx.arc(lensX, lensY, 12, 0, Math.PI * 2)
      ctx.fill()

      // Draw planet orbiting the lens star
      const planetAngle = (lensX * 0.05) % (Math.PI * 2)
      const planetOrbitRadius = 25
      const planetX = lensX + Math.cos(planetAngle) * planetOrbitRadius
      const planetY = lensY + Math.sin(planetAngle) * planetOrbitRadius

      ctx.fillStyle = "#4682b4"
      ctx.beginPath()
      ctx.arc(planetX, planetY, 6, 0, Math.PI * 2)
      ctx.fill()

      // Draw light curve below
      ctx.strokeStyle = "#666"
      ctx.lineWidth = 1
      ctx.beginPath()
      ctx.moveTo(50, 250)
      ctx.lineTo(350, 250)
      ctx.stroke()

      // Draw magnification curve
      const magnification = 1 + lensingStrength
      const curveY = 250 - (magnification - 1) * 40

      ctx.strokeStyle = "#4169e1"
      ctx.lineWidth = 2
      ctx.beginPath()
      ctx.moveTo(50, 250)
      ctx.lineTo(lensX * 0.75 + 50, curveY)
      ctx.stroke()

      // Draw planet spike
      if (distance < 50 && distance > 30) {
        const spikeHeight = 15
        ctx.strokeStyle = "#ff6b4a"
        ctx.lineWidth = 2
        ctx.beginPath()
        ctx.moveTo(lensX * 0.75 + 50, curveY)
        ctx.lineTo(lensX * 0.75 + 50, curveY - spikeHeight)
        ctx.stroke()
      }

      // Labels
      ctx.fillStyle = "#999"
      ctx.font = "12px sans-serif"
      ctx.fillText("Magnification", 10, 230)
      ctx.fillText("Time →", 320, 270)

      // Move lens star
      lensX += 2
      if (lensX > 450) {
        lensX = -50
      }

      requestAnimationFrame(animate)
    }

    const animationId = requestAnimationFrame(animate)

    return () => cancelAnimationFrame(animationId)
  }, [])

  return <canvas ref={canvasRef} className="w-full h-auto max-w-md mx-auto" />
}
