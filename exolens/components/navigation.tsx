"use client"

import Link from "next/link"
import { usePathname } from "next/navigation"
import { Button } from "@/components/ui/button"
import { Telescope } from "lucide-react"
import { cn } from "@/lib/utils"
import { StarField } from "./star-field"

const navItems = [
  { href: "/", label: "Home" },
  { href: "/explorer", label: "3D Explorer" },
  { href: "/methods", label: "Detection Methods" },
  { href: "/laboratory", label: "Planetary Lab" },
  { href: "/about", label: "About" },
]

export function Navigation() {
  const pathname = usePathname()
  console.log(pathname === "/" || pathname === "/explorer" || pathname === "/about")
  return (
    <header className={`fixed h-[65px] top-0 left-0 right-0 z-50 ${ pathname === "/" || pathname === "/explorer" || pathname === "/about" ? "bg-gradient-to-b to-[#0a0a0f] via-[#0f0f1a] from-[#1a1a2e]" : "bg-[#1b1b1b]"} `}>
      {pathname === "/" || pathname === "/explorer" || pathname === "/about" ? <StarField /> : null }
      <nav className="container mx-auto px-4 h-16 flex items-center justify-between">
        {/* Logo */}
        <Link href="/" className="flex items-center gap-2 group">
          <div className="w-8 h-8 rounded-full flex items-center justify-center">
            <Telescope className="w-5 h-5 text-white" />
          </div>
          <span className="text-xl font-bold">
            ExoLab <span className="text-cosmic">3D</span>
          </span>
        </Link>

        {/* Navigation Links */}
        <div className="hidden md:flex items-center gap-1">
          {navItems.map((item) => (
            <Link
              key={item.href}
              href={item.href}
              className={cn(
                "px-4 py-2 rounded-md text-sm font-medium transition-colors duration-200",
                pathname === item.href
                  ? "text-foreground"
                  : "text-muted-foreground hover:text-foreground hover:shadow-[0_0_20px_rgba(65,105,225,0.3),_0_0_40px_rgba(0,217,255,0.2)]",
              )}
            >
              {item.label}
            </Link>
          ))}
        </div>

        {/* CTA Button */}
        <Button asChild size="sm" className="hidden sm:flex transition-all duration-300 hover:shadow-[0_0_20px_rgba(65,105,225,0.3),_0_0_40px_rgba(0,217,255,0.2)]">
          <Link href="/laboratory">Launch Explorer</Link>
        </Button>
      </nav>
    </header>
  )
}
