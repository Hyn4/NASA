import Link from "next/link"
import { Button } from "@/components/ui/button"
import { ExternalLink } from "lucide-react"

export function FeaturedSystem() {
  return (
    <section className="container mx-auto px-4 py-24">
      <div className="max-w-6xl mx-auto">
        <div className="bg-card border border-border rounded-2xl overflow-hidden">
          <div className="grid md:grid-cols-2 gap-0">
            {/* Image side */}
            <div className="relative h-[400px] md:h-auto bg-gradient-to-br from-primary/20 to-accent/20">
              <img src="/artistic-rendering-of-trappist-1-system-with-seven.jpg" alt="TRAPPIST-1 System" className="w-full h-full object-cover" />
              <div className="absolute inset-0 bg-gradient-to-t from-card via-transparent to-transparent" />
            </div>

            {/* Content side */}
            <div className="p-8 md:p-12 flex flex-col justify-center space-y-6">
              <div className="space-y-2">
                <div className="text-sm text-gray-500 font-semibold tracking-wider uppercase">Featured System</div>
                <h2 className="text-4xl md:text-5xl font-bold text-black">TRAPPIST-1</h2>
              </div>

              <p className="text-lg text-muted-foreground leading-relaxed">
                A remarkable system containing seven Earth-sized planets orbiting an ultra-cool red dwarf star. Three
                planets orbit in the habitable zone where liquid water could exist.
              </p>

              <div className="grid grid-cols-3 gap-4 py-4">
                <div className="space-y-1">
                  <div className="text-3xl font-bold text-primary">7</div>
                  <div className="text-sm text-muted-foreground">Planets</div>
                </div>
                <div className="space-y-1">
                  <div className="text-3xl font-bold text-primary">3</div>
                  <div className="text-sm text-muted-foreground">Habitable</div>
                </div>
                <div className="space-y-1">
                  <div className="text-3xl font-bold text-primary">40</div>
                  <div className="text-sm text-muted-foreground">Light Years</div>
                </div>
              </div>

              <Button asChild size="lg" className="w-full sm:w-auto">
                <Link href="/explorer?system=trappist-1">
                  Explore TRAPPIST-1
                  <ExternalLink className="ml-2 w-4 h-4" />
                </Link>
              </Button>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
