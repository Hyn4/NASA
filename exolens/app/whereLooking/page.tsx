export default function WhereLooking() {
  return (
    <div className="w-full min-h-screen bg-black text-white flex flex-col items-center px-6 py-16 space-y-12 gap-15">
      <h1 className="text-5xl font-semibold text-purple-400 text-center mt-20">
        Where should we look?
      </h1>

      <section className="max-w-4xl bg-gray-900/60 backdrop-blur-md rounded-2xl p-8 shadow-lg border border-gray-800">
        <h2 className="text-2xl font-semibold text-cyan-400 mb-4">
            Possible regions to observe exoplanets        
        </h2>
        <p className="text-gray-300 leading-relaxed">
            The universe is vast and full of mysteries.
            Some regions of the sky are particularly promising for the search for exoplanets,
            due to the density of stars and the sensitivity of the instruments used.
            Among them, the following stand out:
        </p>

        <ul className="list-disc list-inside mt-4 space-y-2 text-gray-200">
          <li>
            <span className="text-purple-400 font-medium">Constelação de Lira (Lyra):</span>{" "}
                region observed by the <span className="text-purple-500">Kepler</span> mission,
                where thousands of exoplanets have been detected.
          </li>
          <li>
            <span className="text-purple-400 font-medium">Orion Region:</span>{" "}
            rich in young stars and planetary systems in formation.
          </li>
          <li>
            <span className="text-purple-400 font-medium">Sagittarius Arm:</span>{" "}
            one of the spiral arms of the Milky Way, with a high concentration of Sun-like sta
          </li>
          <li>
            <span className="text-purple-400 font-medium">Solar Neighborhood:</span>{" "}
            nearby stars such as <span className="text-purple-500">Proxima Centauri</span>{" "}
            and <span className="text-purple-500">Tau Ceti</span> are ideal targets for detailed observation.
          </li>
        </ul>
      </section>
    </div>
  );
}
