import { Card } from "@/components/ui/card"
import { TransitAnimation } from "@/components/transit-animation"
import { RadialVelocityAnimation } from "@/components/radial-velocity-animation"
import { MicrolensingAnimation } from "@/components/microlensing-animation"
import { DirectImagingAnimation } from "@/components/direct-imaging-animation"
import { TransitTimingAnimation } from "@/components/transit-timing-animation"

const detectionMethods = [
  {
    name: "Transit Method",
    description:
      "When a planet passes in front of its star from our perspective, it blocks a tiny fraction of the star's light. By measuring this periodic dimming, we can detect the planet and determine its size.",
    discoveries: "~75% of confirmed exoplanets",
    advantages: ["Can determine planet size", "Can detect multiple planets", "Can study atmospheres"],
    component: TransitAnimation,
  },
  {
    name: "Radial Velocity",
    description:
      "As a planet orbits a star, its gravity causes the star to wobble slightly. This wobble creates a Doppler shift in the star's light spectrum, which we can measure to detect the planet.",
    discoveries: "~20% of confirmed exoplanets",
    advantages: ["Can determine planet mass", "Works for non-transiting planets", "Proven technique"],
    component: RadialVelocityAnimation,
  },
  {
    name: "Gravitational Microlensing",
    description:
      "When a star passes in front of a more distant star, its gravity acts as a lens, magnifying the background star's light. If the foreground star has a planet, it creates a distinctive spike in the light curve.",
    discoveries: "~100 confirmed exoplanets",
    advantages: ["Can detect distant planets", "Sensitive to low-mass planets", "Can find free-floating planets"],
    component: MicrolensingAnimation,
  },
  {
    name: "Direct Imaging",
    description:
      "Using powerful telescopes with special techniques to block out the star's light, we can directly photograph planets. This only works for large, young planets far from their stars.",
    discoveries: "~50 confirmed exoplanets",
    advantages: ["Can study planet directly", "Can analyze atmosphere", "Can detect multiple planets"],
    component: DirectImagingAnimation,
  },
  {
    name: "Transit Timing Variations",
    description:
      "If multiple planets orbit a star, their gravitational interactions cause slight variations in the timing of transits. By measuring these variations, we can detect additional planets.",
    discoveries: "~20 confirmed exoplanets",
    advantages: ["Can find non-transiting planets", "Can determine planet masses", "Confirms multi-planet systems"],
    component: TransitTimingAnimation,
  },
]

export function DetectionMethods() {
  return (
    <section>
      <div className="max-w-6xl mx-auto">
        <h2 className="text-3xl md:text-4xl font-bold mb-4 text-center">Detection Methods</h2>
        <p className="text-center text-gray-300 mb-12 max-w-2xl mx-auto">
          Astronomers have developed several ingenious techniques to find planets orbiting distant stars.
        </p>

        <div className="space-y-8">
          {detectionMethods.map((method, index) => (
            <Card key={method.name} className="overflow-hidden border-border !drop-shadow-xl bg-[#1b1b1b]">
              <div className="grid md:grid-cols-2 gap-0">
                {/* Content side */}
                <div className="p-8 space-y-4 order-2 md:order-1">
                  <div className="space-y-2">
                    <div className="text-sm text-accent font-semibold">Method {index + 1}</div>
                    <h3 className="text-2xl md:text-3xl font-bold text-white">{method.name}</h3>
                  </div>

                  <p className="text-gray-300 leading-relaxed">{method.description}</p>

                  <div className="pt-2">
                    <p className="text-sm font-semibold text-foreground mb-2">Discoveries:</p>
                    <p className="text-gray-300 font-bold">{method.discoveries}</p>
                  </div>

                  <div className="pt-2">
                    <p className="text-sm font-semibold text-foreground mb-2">Advantages:</p>
                    <ul className="space-y-1">
                      {method.advantages.map((adv, idx) => (
                        <li key={idx} className="text-sm text-gray-300 flex items-start gap-2">
                          <span className="text-accent mt-1">✓</span>
                          <span>{adv}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>

                {/* Animation side */}
                <div className="bg-black p-8 flex items-center justify-center min-h-[300px] order-1 md:order-2">
                  {method.component ? (
                    <method.component />
                  ) : (
                    <div className="text-center text-gray-300">
                      <p className="text-sm">Animation coming soon</p>
                    </div>
                  )}
                </div>
              </div>
            </Card>
          ))}
        </div>
      </div>
    </section>
  )
}
