'use client'

import { useMemo } from 'react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts'
import type { PlanetSimulationParameters } from './planet-creator'

interface LightCurveSimulationProps {
  parameters: PlanetSimulationParameters
}

export function LightCurveSimulation({ parameters }: LightCurveSimulationProps) {
  // Calcular a curva de luz do trânsito
  const { lightCurveData, transitDepth, transitDuration, maxDepthPercent } = useMemo(() => {
    const { radius } = parameters

    // Constantes
    const EARTH_RADIUS_KM = 6371 // km
    const SUN_RADIUS_KM = 696000 // km

    // Raio do planeta em km
    const planetRadiusKm = radius * EARTH_RADIUS_KM

    // Calcular a profundidade do trânsito
    // Transit depth = (Rp / Rs)²
    const radiusRatio = planetRadiusKm / SUN_RADIUS_KM
    const depth = Math.pow(radiusRatio, 2)
    const depthPpm = depth * 1e6 // em partes por milhão
    const depthPercent = depth * 100 // em porcentagem

    // Duração do trânsito (aproximada para visualização)
    // Para o Sol, vamos assumir ~13 horas como referência (como a Terra)
    const transitDurationHours = 13 * Math.sqrt(radius) // escala com √raio

    // Gerar pontos da curva de luz
    const points = 200
    const data: Array<{ time: number; brightness: number; phase: string }> = []

    // Tempo total da observação (3x a duração do trânsito para contexto)
    const totalTime = transitDurationHours * 3
    const timeStep = totalTime / points

    // Início e fim do trânsito
    const transitStart = totalTime / 3
    const transitEnd = transitStart + transitDurationHours

    for (let i = 0; i < points; i++) {
      const time = i * timeStep
      let brightness = 1.0 // 100% de brilho (normalizado)
      let phase = 'before'

      if (time >= transitStart && time <= transitEnd) {
        // Durante o trânsito
        const transitProgress = (time - transitStart) / transitDurationHours

        // Usar uma função suave para entrada e saída (ingress/egress)
        // Simular o efeito de limb darkening
        let transitFactor = 0

        if (transitProgress < 0.1) {
          // Ingress (entrada) - 10% do tempo
          transitFactor = depth * (transitProgress / 0.1)
          phase = 'ingress'
        } else if (transitProgress > 0.9) {
          // Egress (saída) - 10% do tempo
          transitFactor = depth * ((1 - transitProgress) / 0.1)
          phase = 'egress'
        } else {
          // Trânsito completo (flat bottom)
          transitFactor = depth
          phase = 'transit'
        }

        brightness = 1.0 - transitFactor
      } else if (time > transitEnd) {
        phase = 'after'
      }

      // Adicionar um pouco de ruído realista (variação estelar)
      const noise = (Math.random() - 0.5) * 0.0001
      brightness += noise

      data.push({
        time: parseFloat(time.toFixed(2)),
        brightness: parseFloat((brightness * 100).toFixed(6)), // converter para porcentagem
        phase
      })
    }

    return {
      lightCurveData: data,
      transitDepth: depthPpm,
      transitDuration: transitDurationHours,
      maxDepthPercent: depthPercent
    }
  }, [parameters.radius])

  // Tooltip customizado
  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload
      return (
        <div className="bg-[#1b1b1b] border border-[#404040] p-3 rounded-lg shadow-lg">
          <p className="text-white text-sm font-semibold mb-1">
            Time: {data.time.toFixed(2)} hours
          </p>
          <p className="text-blue-400 text-sm">
            Brightness: {data.brightness.toFixed(4)}%
          </p>
          <p className="text-gray-400 text-xs mt-1 capitalize">
            Phase: {data.phase}
          </p>
        </div>
      )
    }
    return null
  }

  // Calcular limites do eixo Y para melhor visualização
  const minBrightness = Math.min(...lightCurveData.map(d => d.brightness))
  const maxBrightness = 100
  const yAxisPadding = (maxBrightness - minBrightness) * 0.1
  const yMin = Math.max(minBrightness - yAxisPadding, 99.9) // não ir abaixo de 99.9%
  const yMax = 100.05 // pequeno padding no topo

  return (
    <div className="space-y-4">
      {/* Informações do trânsito */}
      <div className="grid grid-cols-2 gap-4 p-4 bg-[#2c2c2c] rounded-lg border border-[#404040]">
        <div>
          <p className="text-xs text-gray-400 mb-1">Transit Depth</p>
          <p className="text-lg font-bold text-white">
            {transitDepth.toFixed(2)} <span className="text-sm font-normal text-gray-400">ppm</span>
          </p>
          <p className="text-xs text-blue-400">
            ({maxDepthPercent.toFixed(4)}% brightness drop)
          </p>
        </div>
        <div>
          <p className="text-xs text-gray-400 mb-1">Transit Duration</p>
          <p className="text-lg font-bold text-white">
            {transitDuration.toFixed(2)} <span className="text-sm font-normal text-gray-400">hours</span>
          </p>
          <p className="text-xs text-gray-400">
            Planet radius: {parameters.radius.toFixed(2)} R⊕
          </p>
        </div>
      </div>

      {/* Gráfico da curva de luz */}
      <div className="w-full h-[300px] bg-[#0a0a0a] rounded-lg p-4 border border-[#333333]">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={lightCurveData} margin={{ top: 10, right: 30, left: 10, bottom: 30 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#333333" />
            <XAxis
              dataKey="time"
              stroke="#9ca3af"
              label={{
                value: 'Time (hours)',
                position: 'insideBottom',
                offset: -10,
                style: { fill: '#9ca3af', fontSize: 12 }
              }}
              tick={{ fill: '#9ca3af', fontSize: 11 }}
            />
            <YAxis
              stroke="#9ca3af"
              domain={[yMin, yMax]}
              tickFormatter={(value) => value.toFixed(3)}
              label={{
                value: 'Relative Brightness (%)',
                angle: -90,
                position: 'insideLeft',
                style: { fill: '#9ca3af', fontSize: 12 }
              }}
              tick={{ fill: '#9ca3af', fontSize: 11 }}
              width={70}
            />
            <Tooltip content={<CustomTooltip />} />

            {/* Linha de referência para 100% */}
            <ReferenceLine
              y={100}
              stroke="#60a5fa"
              strokeDasharray="5 5"
              label={{
                value: 'Baseline (100%)',
                position: 'right',
                fill: '#60a5fa',
                fontSize: 10
              }}
            />

            {/* Linha da curva de luz */}
            <Line
              type="monotone"
              dataKey="brightness"
              stroke="#3b82f6"
              strokeWidth={2}
              dot={false}
              isAnimationActive={true}
              animationDuration={1500}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Explicação */}
      <div className="p-3 bg-[#2c2c2c]/50 rounded-lg border border-[#404040]">
        <p className="text-xs text-gray-400 leading-relaxed">
          <strong className="text-white">Transit Method:</strong> When a planet passes in front of the Sun,
          it blocks a tiny fraction of sunlight. The depth of the dip is proportional to (Planet Radius / Sun Radius)².
          Larger planets create deeper, more detectable transits. This simulation shows the relative brightness
          of the Sun as observed from a distant point.
        </p>
      </div>

      {/* Fases do trânsito */}
      <div className="grid grid-cols-4 gap-2 text-center">
        <div className="p-2 bg-[#2c2c2c] rounded border border-[#404040]">
          <div className="w-3 h-3 bg-gray-600 rounded-full mx-auto mb-1"></div>
          <p className="text-xs text-gray-400">Before</p>
        </div>
        <div className="p-2 bg-[#2c2c2c] rounded border border-yellow-500/30">
          <div className="w-3 h-3 bg-yellow-500 rounded-full mx-auto mb-1"></div>
          <p className="text-xs text-yellow-400">Ingress</p>
        </div>
        <div className="p-2 bg-[#2c2c2c] rounded border border-blue-500/30">
          <div className="w-3 h-3 bg-blue-500 rounded-full mx-auto mb-1"></div>
          <p className="text-xs text-blue-400">Full Transit</p>
        </div>
        <div className="p-2 bg-[#2c2c2c] rounded border border-purple-500/30">
          <div className="w-3 h-3 bg-purple-500 rounded-full mx-auto mb-1"></div>
          <p className="text-xs text-purple-400">Egress</p>
        </div>
      </div>
    </div>
  )
}
