import { NasaDataLoader } from "@/components/nasa-data-loader"

export default function AboutPage() {
  return (
    <div className="min-h-screen pt-16">
      <div className="container mx-auto px-4 py-12">
        <div className="max-w-4xl mx-auto space-y-12">
          <div className="text-center space-y-6">
            <h1 className="text-4xl md:text-6xl font-bold">About ExoVerse</h1>
            <p className="text-xl text-muted-foreground leading-relaxed">
              Transforming complex exoplanet data into immersive, interactive experiences for education and discovery.
            </p>
          </div>

          <div className="prose prose-invert max-w-none space-y-8">
            <section className="space-y-4">
              <h2 className="text-3xl font-bold">Our Mission</h2>
              <p className="text-lg text-muted-foreground leading-relaxed">
                ExoVerse 3D Explorer aims to make the fascinating world of exoplanets accessible to everyone. By
                combining real scientific data from NASA with cutting-edge 3D visualization and AI-generated imagery, we
                create an experience that is both educational and inspiring.
              </p>
            </section>

            <section className="space-y-4">
              <h2 className="text-3xl font-bold">Data Sources</h2>
              <p className="text-lg text-muted-foreground leading-relaxed">
                All planetary data comes from the{" "}
                <a
                  href="https://exoplanetarchive.ipac.caltech.edu/"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-primary hover:text-accent transition-colors"
                >
                  NASA Exoplanet Archive
                </a>
                , which maintains the most comprehensive database of confirmed exoplanets discovered through various
                detection methods. The archive is updated regularly as new discoveries are made.
              </p>
            </section>

            <section className="space-y-4">
              <h2 className="text-3xl font-bold">Technology</h2>
              <p className="text-lg text-muted-foreground leading-relaxed">
                Built with Next.js, Three.js for 3D visualization, and AI image generation to create realistic planetary
                textures based on actual physical parameters. The NASA Exoplanet Archive API provides real-time access
                to the latest exoplanet discoveries and data.
              </p>
            </section>

            <section className="space-y-4">
              <h2 className="text-3xl font-bold">AI-Generated Textures</h2>
              <p className="text-lg text-muted-foreground leading-relaxed">
                Our system uses AI to generate photorealistic planet textures based on scientific parameters like
                temperature, size, composition, and star type. Each planet's appearance is uniquely generated to reflect
                its actual physical characteristics, making the visualization both beautiful and scientifically
                grounded.
              </p>
            </section>
          </div>

          {/* NASA Data Demo */}
          <div className="pt-8">
            <h2 className="text-2xl font-bold mb-6 text-center">Live NASA Data Integration</h2>
            <NasaDataLoader />
          </div>
        </div>
      </div>
    </div>
  )
}
