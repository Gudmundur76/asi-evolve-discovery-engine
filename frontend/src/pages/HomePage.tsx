import { useMemo } from 'react';
import { FlaskConical, TrendingDown, Target, Zap } from 'lucide-react';
import { Header } from '../components/Header';
import { DiscoveryFeed } from '../components/DiscoveryFeed';
import { LoopStatusPanel } from '../components/LoopStatusPanel';
import { useWebSocket } from '../hooks/useWebSocket';
import { useFetch } from '../hooks/useApi';
import type { Discovery, LoopStatus } from '../types';

export function HomePage() {
  const { status, lastCycle, connected, sendMessage } = useWebSocket();
  const { data: discoveries } = useFetch<Discovery[]>('/api/discoveries');

  const stats = useMemo(() => {
    if (!discoveries || discoveries.length === 0) {
      return {
        total: 0,
        passed: 0,
        bestAffinity: 0,
        avgConfidence: 0,
      };
    }

    const passed = discoveries.filter((d) => d.overall_passed).length;
    const bestAffinity = Math.min(...discoveries.map((d) => d.predicted_affinity_nm));
    const avgConfidence =
      discoveries.reduce((sum, d) => sum + d.confidence_score, 0) / discoveries.length;

    return {
      total: discoveries.length,
      passed,
      bestAffinity,
      avgConfidence,
    };
  }, [discoveries]);

  const handleStart = () => {
    sendMessage({ action: 'start_loop' });
  };

  const handleStop = () => {
    sendMessage({ action: 'stop_loop' });
  };

  const handleStep = () => {
    sendMessage({ action: 'step_loop' });
  };

  return (
    <div className="min-h-screen bg-navy-900">
      <Header loopRunning={status?.running ?? false} />

      {/* Hero Section */}
      <section className="relative border-b border-navy-700/50 overflow-hidden">
        {/* Background decoration */}
        <div className="absolute inset-0 overflow-hidden pointer-events-none">
          <div className="absolute -top-20 -right-20 w-80 h-80 bg-accent/5 rounded-full blur-3xl" />
          <div className="absolute -bottom-10 -left-10 w-60 h-60 bg-accent/3 rounded-full blur-3xl" />
        </div>

        <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12 md:py-16">
          <div className="flex items-center gap-3 mb-4">
            <FlaskConical className="w-8 h-8 text-accent" />
            <span className="text-sm font-medium text-accent tracking-wider uppercase">
              Molecular Discovery Engine
            </span>
          </div>
          <h1 className="text-3xl md:text-4xl lg:text-5xl font-bold text-text-primary mb-4 tracking-tight">
            ASI-Evolve <span className="text-gradient">Discovery Engine</span>
          </h1>
          <p className="text-base md:text-lg text-text-secondary max-w-2xl mb-8 leading-relaxed">
            Autonomous molecular discovery powered by iterative agent loops, predictive modeling,
            and real-time ADMET evaluation. Accelerating drug candidate identification from
            weeks to hours.
          </p>

          {/* Live Stats */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 md:gap-4">
            <div className="bg-navy-800/80 border border-navy-700/50 rounded-card p-4">
              <div className="flex items-center gap-2 mb-2">
                <FlaskConical className="w-4 h-4 text-accent" />
                <span className="text-xs text-text-secondary uppercase tracking-wider">Total</span>
              </div>
              <span className="text-2xl font-mono font-semibold text-text-primary">
                {stats.total.toLocaleString()}
              </span>
            </div>
            <div className="bg-navy-800/80 border border-navy-700/50 rounded-card p-4">
              <div className="flex items-center gap-2 mb-2">
                <Target className="w-4 h-4 text-success" />
                <span className="text-xs text-text-secondary uppercase tracking-wider">Passed</span>
              </div>
              <span className="text-2xl font-mono font-semibold text-success">
                {stats.passed.toLocaleString()}
              </span>
            </div>
            <div className="bg-navy-800/80 border border-navy-700/50 rounded-card p-4">
              <div className="flex items-center gap-2 mb-2">
                <TrendingDown className="w-4 h-4 text-warning" />
                <span className="text-xs text-text-secondary uppercase tracking-wider">Best nM</span>
              </div>
              <span className="text-2xl font-mono font-semibold text-warning">
                {stats.bestAffinity < 0.001 && stats.total > 0
                  ? '<0.001'
                  : stats.bestAffinity.toFixed(3)}
              </span>
            </div>
            <div className="bg-navy-800/80 border border-navy-700/50 rounded-card p-4">
              <div className="flex items-center gap-2 mb-2">
                <Zap className="w-4 h-4 text-accent" />
                <span className="text-xs text-text-secondary uppercase tracking-wider">Confidence</span>
              </div>
              <span className="text-2xl font-mono font-semibold text-accent">
                {(stats.avgConfidence * 100).toFixed(0)}%
              </span>
            </div>
          </div>
        </div>
      </section>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="flex flex-col lg:flex-row gap-6">
          {/* Left: Discovery Feed (2/3) */}
          <div className="flex-1 lg:w-2/3 min-w-0">
            <DiscoveryFeed />
          </div>

          {/* Right: Loop Status Panel (1/3) */}
          <div className="lg:w-80 xl:w-96 flex-shrink-0">
            <div className="lg:sticky lg:top-20">
              <LoopStatusPanel
                status={status ?? ({} as LoopStatus)}
                lastCycle={lastCycle}
                connected={connected}
                onStart={handleStart}
                onStop={handleStop}
                onStep={handleStep}
              />
            </div>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-navy-700/50 mt-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex flex-col sm:flex-row items-center justify-between gap-4 text-xs text-text-secondary">
            <div className="flex items-center gap-2">
              <FlaskConical className="w-4 h-4 text-accent" />
              <span>
                ASI-Evolve Discovery Engine &copy; {new Date().getFullYear()}
              </span>
            </div>
            <div className="flex items-center gap-4">
              <span>Powered by ChEMBL, PDB, SwissADME</span>
              <span className="hidden sm:inline">|</span>
              <span className="hidden sm:inline">React + TypeScript + Tailwind</span>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}
