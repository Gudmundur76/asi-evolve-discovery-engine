import { Link, useLocation } from 'react-router-dom';
import { FlaskConical, Menu, X, Activity } from 'lucide-react';
import { useState } from 'react';

interface HeaderProps {
  loopRunning?: boolean;
}

export function Header({ loopRunning = false }: HeaderProps) {
  const location = useLocation();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  const isActive = (path: string) => location.pathname === path;

  const navLinks = [
    { path: '/', label: 'Discoveries' },
    { path: '/about', label: 'About' },
  ];

  return (
    <header className="sticky top-0 z-50 bg-navy-900/90 backdrop-blur-md border-b border-navy-700/50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <Link to="/" className="flex items-center gap-3 group">
            <FlaskConical className="w-7 h-7 text-accent group-hover:rotate-12 transition-transform" />
            <span className="font-bold text-lg tracking-tight">
              <span className="text-text-primary">ASI-</span>
              <span className="text-accent">Evolve</span>
            </span>
          </Link>

          {/* Desktop Navigation */}
          <nav className="hidden md:flex items-center gap-1">
            {navLinks.map((link) => (
              <Link
                key={link.path}
                to={link.path}
                className={`px-4 py-2 text-sm font-medium rounded-card transition-all duration-200 ${
                  isActive(link.path)
                    ? 'text-accent bg-accent/10'
                    : 'text-text-secondary hover:text-text-primary hover:bg-navy-800'
                }`}
              >
                {link.label}
              </Link>
            ))}
          </nav>

          {/* Loop Status + Mobile Toggle */}
          <div className="flex items-center gap-4">
            {/* Loop Status Indicator */}
            <div className="flex items-center gap-2 px-3 py-1.5 bg-navy-800 rounded-full border border-navy-700">
              <Activity
                className={`w-3.5 h-3.5 ${
                  loopRunning ? 'text-success' : 'text-text-secondary'
                }`}
              />
              <span
                className={`w-2 h-2 rounded-full ${
                  loopRunning
                    ? 'bg-success animate-pulse-slow'
                    : 'bg-text-secondary'
                }`}
              />
              <span className="text-xs font-medium text-text-secondary hidden sm:inline">
                {loopRunning ? 'Running' : 'Idle'}
              </span>
            </div>

            {/* Mobile Menu Toggle */}
            <button
              className="md:hidden p-2 text-text-secondary hover:text-text-primary transition-colors"
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              aria-label="Toggle menu"
            >
              {mobileMenuOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
            </button>
          </div>
        </div>

        {/* Mobile Navigation */}
        {mobileMenuOpen && (
          <nav className="md:hidden pb-4 border-t border-navy-700/50 pt-3 animate-fade-in">
            {navLinks.map((link) => (
              <Link
                key={link.path}
                to={link.path}
                onClick={() => setMobileMenuOpen(false)}
                className={`block px-4 py-2.5 text-sm font-medium rounded-card transition-all ${
                  isActive(link.path)
                    ? 'text-accent bg-accent/10'
                    : 'text-text-secondary hover:text-text-primary hover:bg-navy-800'
                }`}
              >
                {link.label}
              </Link>
            ))}
          </nav>
        )}
      </div>
    </header>
  );
}
