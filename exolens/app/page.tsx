import { Rocket, Telescope, Sparkles } from "lucide-react"
import { StarField } from "@/components/star-field"
import { HeroSection } from "@/components/hero-section"
import { FeaturedSystem } from "@/components/featured-system"

export default function HomePage() {
  return (
    <div className="relative min-h-screen">
      <StarField />

      <main className="relative z-10">
        <HeroSection />
        <FeaturedSystem />

        {/* Mission Section */}
        <section className="container mx-auto px-4 py-24">
          <div className="max-w-4xl mx-auto text-center space-y-6">
            <h2 className="text-4xl md:text-5xl font-bold text-balance">
              Transforming Data Into <span className="text-cosmic">Discovery</span>
            </h2>
            <p className="text-xl text-muted-foreground leading-relaxed">
              ExoVerse 3D Explorer transforms complex exoplanet data from NASA into an immersive, interactive
              experience. Explore over 6,000 confirmed exoplanets, understand detection methods, and create your own
              worlds.
            </p> 
          </div>

          {/* Feature Cards */}q
          <div className="grid md:grid-cols-3 gap-8 mt-16 max-w-6xl mx-auto">
            <div className="bg-card border border-border rounded-lg p-8 space-y-4 hover:border-primary transition-colors">
              <div className="w-12 h-12 rounded-full bg-primary/10 flex items-center justify-center">
                <Telescope className="w-6 h-6 text-primary" />
              </div>
              <h3 className="text-2xl font-bold text-black">3D Exploration</h3>
              <p className="text-muted-foreground leading-relaxed">
                Navigate real planetary systems in immersive 3D. Interact with planets based on actual NASA data.
              </p>
            </div>

            <div className="bg-card border border-border rounded-lg p-8 space-y-4 hover:border-accent transition-colors">
              <div className="w-12 h-12 rounded-full bg-primary/10 flex items-center justify-center">
                <Sparkles className="w-6 h-6 text-primary" />
              </div>
              <h3 className="text-2xl font-bold text-black">AI-Generated Worlds</h3>
              <p className="text-muted-foreground leading-relaxed">
                See planets come to life with AI-generated textures based on their real physical properties.
              </p>
            </div>

            <div className="bg-card border border-border rounded-lg p-8 space-y-4 hover:border-primary transition-colors">
              <div className="w-12 h-12 rounded-full bg-primary/10 flex items-center justify-center">
                <Rocket className="w-6 h-6 text-primary" />
              </div>
              <h3 className="text-2xl font-bold text-black">Learn & Create</h3>
              <p className="text-muted-foreground leading-relaxed">
                Understand detection methods and design your own planets in the Planetary Laboratory.
              </p>
            </div>
          </div>
        </section>
      </main>
    </div>
  )
}
