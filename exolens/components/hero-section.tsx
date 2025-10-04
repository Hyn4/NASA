import Link from "next/link"
import { Button } from "@/components/ui/button"
import { ArrowRight } from "lucide-react"

export function HeroSection() {
  return (
    <section className="container mx-auto px-4 pt-32 pb-24">
      <div className="max-w-5xl mx-auto text-center space-y-8">
        {/* Animated badge */}
        <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-primary/10 border border-primary/20 text-sm">
          <span className="relative flex h-2 w-2">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-accent opacity-75"></span>
            <span className="relative inline-flex rounded-full h-2 w-2 bg-accent"></span>
          </span>
          <span className="text-foreground">6,000+ Confirmed Exoplanets</span>
        </div>

        {/* Main heading */}
        <h1 className="text-5xl md:text-7xl lg:text-8xl font-bold text-balance leading-tight">
          Explore Worlds <span className="text-cosmic">Beyond</span>
          <br />
          Our Solar System
        </h1>

        {/* Subtitle */}
        <p className="text-xl md:text-2xl text-muted-foreground max-w-3xl mx-auto leading-relaxed">
          An interactive journey through real exoplanet data. Visualize distant worlds, understand how we find them, and
          create your own planets.
        </p>

        {/* CTA Buttons */}
        <div className="flex flex-col sm:flex-row gap-4 justify-center items-center pt-4">
          <Button asChild size="lg" className="text-lg px-8 py-6 cosmic-glow">
            <Link href="/explorer">
              Start Exploring
              <ArrowRight className="ml-2 w-5 h-5" />
            </Link>
          </Button>

          <Button
            asChild
            size="lg"
            variant="outline"
            className="text-lg px-8 py-6 border-border hover:border-primary bg-transparent"
          >
            <Link href="/methods">How We Find Planets</Link>
          </Button>
        </div>
      </div>
    </section>
  )
}
