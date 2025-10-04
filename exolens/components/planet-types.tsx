import { Card } from "@/components/ui/card"

const planetTypes = [
  {
    name: "Gas Giant",
    description:
      "Massive planets composed primarily of hydrogen and helium, similar to Jupiter and Saturn. These are the easiest to detect due to their large size.",
    characteristics: ["Mass > 100 Earth masses", "Thick atmosphere", "No solid surface"],
    color: "from-orange-500 to-amber-600",
    image: "/gas-giant-exoplanet-jupiter-like.jpg",
  },
  {
    name: "Neptune-like",
    description:
      "Ice giants with thick atmospheres of hydrogen, helium, and methane. Smaller than gas giants but still much larger than Earth.",
    characteristics: ["10-100 Earth masses", "Icy composition", "Thick atmosphere"],
    color: "from-blue-500 to-cyan-600",
    image: "/neptune-like-ice-giant-exoplanet.jpg",
  },
  {
    name: "Super-Earth",
    description:
      "Rocky planets larger than Earth but smaller than Neptune. They may have thick atmospheres or be rocky worlds with extreme conditions.",
    characteristics: ["1.5-10 Earth masses", "Rocky or icy", "Potentially habitable"],
    color: "from-green-500 to-emerald-600",
    image: "/super-earth-rocky-exoplanet.jpg",
  },
  {
    name: "Terrestrial",
    description:
      "Earth-sized rocky planets with solid surfaces. These are the most interesting for the search for life but also the hardest to detect.",
    characteristics: ["< 1.5 Earth masses", "Rocky surface", "Thin atmosphere"],
    color: "from-blue-600 to-indigo-700",
    image: "/terrestrial-earth-like-exoplanet.jpg",
  },
]

export function PlanetTypes() {
  return (
    <section className="mb-24">
      <div className="max-w-6xl mx-auto">
        <h2 className="text-3xl md:text-4xl font-bold mb-4 text-center">Types of Exoplanets</h2>
        <p className="text-center text-gray-300 mb-12 max-w-2xl mx-auto">
          Exoplanets come in many varieties, from scorching hot Jupiters to potentially habitable rocky worlds.
        </p>

        <div className="grid md:grid-cols-2 gap-8">
          {planetTypes.map((type) => (
            <Card key={type.name} className="overflow-hidden !drop-shadow-xl transition-colors bg-[#1b1b1b]">
              <div className={`h-48 bg-gradient-to-br ${type.color} relative`}>
                <img
                  src={type.image || "/placeholder.svg"}
                  alt={type.name}
                  className="w-full h-full object-cover mix-blend-overlay"
                />
              </div>
              <div className="p-6 space-y-4">
                <h3 className="text-2xl font-bold text-white">{type.name}</h3>
                <p className="text-gray-300 leading-relaxed">{type.description}</p>
                <div className="space-y-2">
                  <p className="text-sm font-semibold text-gray-300">Key Characteristics:</p>
                  <ul className="space-y-1">
                    {type.characteristics.map((char, idx) => (
                      <li key={idx} className="text-sm text-gray-300 flex items-start gap-2">
                        <span className="text-primary mt-1">•</span>
                        <span>{char}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              </div>
            </Card>
          ))}
        </div>
      </div>
    </section>
  )
}
