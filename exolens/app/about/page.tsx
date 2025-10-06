import { NasaDataLoader } from "@/components/nasa-data-loader"
import { Rocket, Database, Brain, Globe, Users, Award, Telescope, Sparkles } from "lucide-react"
import { Card } from "@/components/ui/card"

export default function AboutPage() {
  return (
    <div className="min-h-screen pt-16 bg-gradient-to-b from-black via-[#0a0a0a] to-black">
      <div className="container mx-auto px-4 py-12">
        <div className="max-w-6xl mx-auto space-y-16">

          {/* Hero Section */}
          <div className="text-center space-y-6 py-8">
            <div className="inline-flex items-center gap-2 px-4 py-2 bg-blue-600/20 border border-blue-500/30 rounded-full text-blue-400 text-sm font-medium mb-4">
              <Sparkles className="w-4 h-4" />
              NASA Space Apps Challenge 2025
            </div>
            <h1 className="text-5xl md:text-7xl font-bold bg-gradient-to-r from-blue-400 via-purple-400 to-pink-400 bg-clip-text text-transparent">
              About ExoLab
            </h1>
            <p className="text-xl md:text-2xl text-gray-300 leading-relaxed max-w-3xl mx-auto">
              Transforming complex exoplanet data into immersive, interactive experiences for education and discovery
            </p>
          </div>

          {/* The Challenge Section */}
          <section className="space-y-6">
            <div className="flex items-center gap-3 mb-6">
              <div className="p-3 bg-red-600/20 rounded-lg border border-red-500/30">
                <Telescope className="w-6 h-6 text-red-400" />
              </div>
              <h2 className="text-3xl md:text-4xl font-bold text-white">The Challenge</h2>
            </div>
            <Card className="p-8 bg-[#1b1b1b] border-[#333333] shadow-2xl">
              <p className="text-lg text-gray-300 leading-relaxed mb-4">
                The universe of exoplanets has grown rapidly, with NASA confirming over <strong className="text-blue-400">6,000 worlds</strong> beyond our Solar System. Despite this acceleration in discoveries—driven primarily by the <strong className="text-purple-400">Transiting Exoplanet Survey Satellite (TESS)</strong> and previous missions like <strong className="text-purple-400">Kepler</strong>—detailed scientific information about these worlds remains technical, fragmented, and difficult to visualize for the general public and students.
              </p>
              <p className="text-lg text-gray-300 leading-relaxed">
                <strong className="text-white">ExoVerse bridges this gap</strong>, transforming raw and complex data into an interactive and accessible journey, fulfilling the goal of increasing public engagement and understanding of exoplanet science.
              </p>
            </Card>
          </section>

          {/* Our Solution Section */}
          <section className="space-y-6">
            <div className="flex items-center gap-3 mb-6">
              <div className="p-3 bg-green-600/20 rounded-lg border border-green-500/30">
                <Globe className="w-6 h-6 text-green-400" />
              </div>
              <h2 className="text-3xl md:text-4xl font-bold text-white">Our Solution</h2>
            </div>
            <div className="grid md:grid-cols-3 gap-6">
              <Card className="p-6 bg-[#1b1b1b] border-[#333333] hover:border-blue-500/50 transition-all duration-300">
                <div className="p-3 bg-blue-600/20 rounded-lg border border-blue-500/30 w-fit mb-4">
                  <Rocket className="w-8 h-8 text-blue-400" />
                </div>
                <h3 className="text-xl font-bold text-white mb-3">3D Visualization</h3>
                <p className="text-gray-400 leading-relaxed">
                  Real-time 3D rendering of exoplanets with scientifically accurate parameters, orbital mechanics, and host star interactions.
                </p>
              </Card>

              <Card className="p-6 bg-[#1b1b1b] border-[#333333] hover:border-purple-500/50 transition-all duration-300">
                <div className="p-3 bg-purple-600/20 rounded-lg border border-purple-500/30 w-fit mb-4">
                  <Brain className="w-8 h-8 text-purple-400" />
                </div>
                <h3 className="text-xl font-bold text-white mb-3">AI Generation</h3>
                <p className="text-gray-400 leading-relaxed">
                  Machine learning models generate photorealistic planet textures based on temperature, composition, size, and stellar characteristics.
                </p>
              </Card>

              <Card className="p-6 bg-[#1b1b1b] border-[#333333] hover:border-green-500/50 transition-all duration-300">
                <div className="p-3 bg-green-600/20 rounded-lg border border-green-500/30 w-fit mb-4">
                  <Database className="w-8 h-8 text-green-400" />
                </div>
                <h3 className="text-xl font-bold text-white mb-3">Real NASA Data</h3>
                <p className="text-gray-400 leading-relaxed">
                  Live integration with NASA Exoplanet Archive, updated with the latest discoveries from TESS, Kepler, and JWST missions.
                </p>
              </Card>
            </div>
          </section>

          {/* Data Sources Section */}
          <section className="space-y-6">
            <div className="flex items-center gap-3 mb-6">
              <div className="p-3 bg-blue-600/20 rounded-lg border border-blue-500/30">
                <Database className="w-6 h-6 text-blue-400" />
              </div>
              <h2 className="text-3xl md:text-4xl font-bold text-white">Data Sources</h2>
            </div>

            <Card className="p-8 bg-[#1b1b1b] border-[#333333] shadow-2xl space-y-6">
              <p className="text-lg text-gray-300 leading-relaxed">
                The <a href="https://exoplanetarchive.ipac.caltech.edu/" target="_blank" rel="noopener noreferrer" className="text-blue-400 hover:text-blue-300 underline transition-colors">NASA Exoplanet Archive</a> serves as our foundation, providing the comprehensive database of confirmed planets. We integrate data from multiple transit detection missions, including:
              </p>

              <div className="grid md:grid-cols-2 gap-4">
                <div className="p-5 bg-[#2c2c2c] rounded-lg border border-[#404040]">
                  <h4 className="text-lg font-bold text-white mb-2">🛰️ TESS (Transiting Exoplanet Survey Satellite)</h4>
                  <p className="text-sm text-gray-400">
                    Near real-time exoplanet candidates (TOIs - TESS Objects of Interest) discovered by surveying the brightest stars near Earth.
                  </p>
                </div>

                <div className="p-5 bg-[#2c2c2c] rounded-lg border border-[#404040]">
                  <h4 className="text-lg font-bold text-white mb-2">🔭 Kepler & K2 Missions</h4>
                  <p className="text-sm text-gray-400">
                    Historical catalogs providing essential time-series data and thousands of confirmed exoplanets from deep space surveys.
                  </p>
                </div>

                <div className="p-5 bg-[#2c2c2c] rounded-lg border border-[#404040] md:col-span-2">
                  <h4 className="text-lg font-bold text-white mb-2">🌌 JWST (James Webb Space Telescope)</h4>
                  <p className="text-sm text-gray-400 mb-3">
                    Metadata and target information from the most powerful space observatory ever built. JWST is crucial for exploring distant worlds and characterizing the potential for life on planets around other stars.
                  </p>
                  <p className="text-xs text-gray-500 italic">
                    Note: Canada contributed the NIRISS scientific instrument and FGS guidance sensor to JWST, enabling groundbreaking exoplanet atmospheric studies.
                  </p>
                </div>
              </div>

              <div className="pt-4 border-t border-[#404040]">
                <p className="text-sm text-gray-400">
                  All data is regularly updated as new discoveries are validated and published by the astronomical community.
                </p>
              </div>
            </Card>
          </section>

          {/* Technology Stack Section */}
          <section className="space-y-6">
            <div className="flex items-center gap-3 mb-6">
              <div className="p-3 bg-purple-600/20 rounded-lg border border-purple-500/30">
                <Sparkles className="w-6 h-6 text-purple-400" />
              </div>
              <h2 className="text-3xl md:text-4xl font-bold text-white">Technology Stack</h2>
            </div>

            <div className="grid md:grid-cols-2 gap-6">
              <Card className="p-6 bg-[#1b1b1b] border-[#333333]">
                <h3 className="text-xl font-bold text-white mb-4">Frontend & 3D</h3>
                <ul className="space-y-3">
                  <li className="flex items-start gap-3">
                    <span className="text-blue-400 mt-1">▹</span>
                    <div>
                      <strong className="text-white">Next.js </strong>
                      <p className="text-sm text-gray-400">React framework with server-side rendering and optimal performance</p>
                    </div>
                  </li>
                  <li className="flex items-start gap-3">
                    <span className="text-blue-400 mt-1">▹</span>
                    <div>
                      <strong className="text-white">Three.js & React Three Fiber</strong>
                      <p className="text-sm text-gray-400">Real-time 3D graphics rendering with WebGL acceleration</p>
                    </div>
                  </li>
                  <li className="flex items-start gap-3">
                    <span className="text-blue-400 mt-1">▹</span>
                    <div>
                      <strong className="text-white">TypeScript</strong>
                      <p className="text-sm text-gray-400">Type-safe development for robust code quality</p>
                    </div>
                  </li>
                </ul>
              </Card>

              <Card className="p-6 bg-[#1b1b1b] border-[#333333]">
                <h3 className="text-xl font-bold text-white mb-4">Backend & AI</h3>
                <ul className="space-y-3">
                  <li className="flex items-start gap-3">
                    <span className="text-purple-400 mt-1">▹</span>
                    <div>
                      <strong className="text-white">Python & FastAPI</strong>
                      <p className="text-sm text-gray-400">High-performance API for data processing and ML inference</p>
                    </div>
                  </li>
                  <li className="flex items-start gap-3">
                    <span className="text-purple-400 mt-1">▹</span>
                    <div>
                      <strong className="text-white">Machine Learning Models</strong>
                      <p className="text-sm text-gray-400">Neural networks for exoplanet classification and texture generation</p>
                    </div>
                  </li>
                  <li className="flex items-start gap-3">
                    <span className="text-purple-400 mt-1">▹</span>
                    <div>
                      <strong className="text-white">NASA Exoplanet Archive API</strong>
                      <p className="text-sm text-gray-400">Real-time access to validated exoplanet data and discoveries</p>
                    </div>
                  </li>
                </ul>
              </Card>
            </div>
          </section>

          {/* Our Team Section */}
          <section className="space-y-6">
            <div className="flex items-center gap-3 mb-6">
              <div className="p-3 bg-yellow-600/20 rounded-lg border border-yellow-500/30">
                <Users className="w-6 h-6 text-yellow-400" />
              </div>
              <h2 className="text-3xl md:text-4xl font-bold text-white">Our Team: Cosmic Hackers</h2>
            </div>

            <Card className="p-8 bg-gradient-to-br from-[#1b1b1b] via-[#1f1f1f] to-[#1b1b1b] border-[#333333] shadow-2xl">
              <div className="space-y-6">
                <div className="flex items-center gap-3 mb-6">
                  <div className="h-1 w-12 bg-gradient-to-r from-blue-500 to-purple-500 rounded"></div>
                  <p className="text-lg text-gray-300">
                    <strong className="text-white">São José do Rio Preto, São Paulo, Brazil</strong>
                  </p>
                </div>

                <p className="text-lg text-gray-300 leading-relaxed">
                  We are a multidisciplinary team of developers and data scientists from São José do Rio Preto (SP), united by the mission to use technology to democratize space science and inspire the next generation of scientists.
                </p>

                <div className="grid md:grid-cols-5 gap-4 pt-6">
                  {[
                    { name: "Mateus Baroni", role: "Full Stack Developer" },
                    { name: "Luan Batista", role: "AI/ML Engineer" },
                    { name: "Yhan Pena", role: "3D Graphics Developer" },
                    { name: "Leandro Aguiar", role: "Data Scientist" },
                    { name: "Otavio Teixeira", role: "Backend Engineer" }
                  ].map((member, idx) => (
                    <div key={idx} className="p-4 bg-[#2c2c2c] rounded-lg border border-[#404040] text-center hover:border-blue-500/50 transition-all duration-300">
                      <div className="w-12 h-12 bg-gradient-to-br from-blue-500 to-purple-500 rounded-full mx-auto mb-3 flex items-center justify-center text-white font-bold text-lg">
                        {member.name.charAt(0)}
                      </div>
                      <h4 className="text-white font-semibold text-sm mb-1">{member.name}</h4>
                      <p className="text-xs text-gray-400">{member.role}</p>
                    </div>
                  ))}
                </div>

                <div className="pt-6 border-t border-[#404040]">
                  <h4 className="text-lg font-bold text-white mb-3">Our Expertise</h4>
                  <div className="grid md:grid-cols-2 gap-4">
                    <div className="flex items-start gap-3">
                      <span className="text-blue-400 text-2xl">🤖</span>
                      <div>
                        <strong className="text-white text-sm">Data Science & AI</strong>
                        <p className="text-xs text-gray-400">Processing physical parameters (temperature, size, stellar type) and generating photorealistic planet textures with scientific accuracy.</p>
                      </div>
                    </div>
                    <div className="flex items-start gap-3">
                      <span className="text-purple-400 text-2xl">💻</span>
                      <div>
                        <strong className="text-white text-sm">Immersive Development</strong>
                        <p className="text-xs text-gray-400">Building ExoVerse 3D Explorer with Next.js and Three.js, ensuring engaging and high-quality interactive visualization.</p>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </Card>
          </section>

          {/* Mission & Awards Section */}
          <section className="space-y-6">
            <div className="flex items-center gap-3 mb-6">
              <div className="p-3 bg-pink-600/20 rounded-lg border border-pink-500/30">
                <Award className="w-6 h-6 text-pink-400" />
              </div>
              <h2 className="text-3xl md:text-4xl font-bold text-white">Our Mission & Goals</h2>
            </div>

            <Card className="p-8 bg-[#1b1b1b] border-[#333333] shadow-2xl">
              <p className="text-lg text-gray-300 leading-relaxed mb-6">
                Our goal is to demonstrate the potential of <strong className="text-blue-400">NASA's open science</strong> initiative and inspire the next generation of scientists by making exoplanet research accessible, engaging, and educational for everyone.
              </p>

              <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4">
                {/* Best Use of Science */}
                <div className="p-5 bg-gradient-to-br from-indigo-600/10 to-blue-600/10 rounded-lg border border-indigo-500/30">
                  <h4 className="text-lg font-bold text-white mb-2">🏆 Best Use of Science</h4>
                  <p className="text-sm text-gray-400">
                    The project that makes the best and most valid use of science and/or the scientific method.
                  </p>
                </div>

                {/* Best Use of Data */}
                <div className="p-5 bg-gradient-to-br from-yellow-600/10 to-orange-600/10 rounded-lg border border-yellow-500/30">
                  <h4 className="text-lg font-bold text-white mb-2">🏆 Best Use of Data</h4>
                  <p className="text-sm text-gray-400">
                    The project that best makes space data accessible or leverages it for a unique application.
                  </p>
                </div>

                {/* Best Use of Technology */}
                <div className="p-5 bg-gradient-to-br from-blue-600/10 to-purple-600/10 rounded-lg border border-blue-500/30">
                  <h4 className="text-lg font-bold text-white mb-2">🏆 Best Use of Technology</h4>
                  <p className="text-sm text-gray-400">
                    Leveraging cutting-edge 3D rendering, AI generation, and real-time data integration to create an unprecedented educational experience.
                  </p>
                </div>

                {/* Galactic Impact */}
                <div className="p-5 bg-gradient-to-br from-green-600/10 to-emerald-600/10 rounded-lg border border-green-500/30">
                  <h4 className="text-lg font-bold text-white mb-2">🏆 Galactic Impact</h4>
                  <p className="text-sm text-gray-400">
                    The project with the most potential to improve life on Earth or in the universe.
                  </p>
                </div>
              </div>
            </Card>
          </section>

          {/* NASA Data Demo */}
          <section className="pt-8">
            <div className="flex items-center gap-3 mb-6">
              <div className="p-3 bg-green-600/20 rounded-lg border border-green-500/30">
                <Database className="w-6 h-6 text-green-400" />
              </div>
              <h2 className="text-3xl md:text-4xl font-bold text-white">Live NASA Data Integration</h2>
            </div>
            <NasaDataLoader />
          </section>
        </div>
      </div>
    </div>
  )
}
