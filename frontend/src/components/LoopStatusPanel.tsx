import { useState } from 'react';
import {
  Play,
  Square,
  StepForward,
  ChevronDown,
  ChevronUp,
  Activity,
  Target,
  TrendingDown,
  Clock,
  Zap,
  BookOpen,
} from 'lucide-react';
import type { LoopStatus, CycleResult } from '../types';

interface LoopStatusPanelProps {
  status: LoopStatus | null;
  lastCycle: CycleResult | null;
  connected: boolean;
  onStart?: () => void;
  onStop?: () => void;
  onStep?: () => void;
}

export function LoopStatusPanel({
  status,
  lastCycle,
  connected,
  onStart,
  onStop,
  onStep,
}: LoopStatusPanelProps) {
  const [collapsed, setCollapsed] = useState(false);

  const cycleCount = status?.cycle_count ?? 0;
  const bestAffinity = status?.current_best_affinity ?? 0;
  const isRunning = status?.running ?? false;
  const targetName = status?.target_name ?? '—';

  // Simulated lessons from cycle results
  const lessons: string[] = [];
  if (lastCycle?.lesson) {
    lessons.push(lastCycle.lesson);
  }
  // Fill with placeholder lessons if empty
  if (lessons.length === 0) {
    lessons.push('No lessons recorded yet. Start the loop to begin learning.');
  }

  return (
    <div className="card-surface overflow-hidden">
      {/* Panel Header */}
      <button
        onClick={() => setCollapsed(!collapsed)}
        className="w-full flex items-center justify-between p-4 hover:bg-navy-700/30 transition-colors"
      >
        <div className="flex items-center gap-3">
          <Activity className={`w-5 h-5 ${isRunning ? 'text-success' : 'text-text-secondary'}`} />
          <h2 className="font-semibold text-text-primary">Loop Status</h2>
          {/* Connection indicator */}
          <span
            className={`w-2 h-2 rounded-full ${connected ? 'bg-success' : 'bg-danger'}`}
            title={connected ? 'Connected' : 'Disconnected'}
          />
        </div>
        {collapsed ? (
          <ChevronDown className="w-4 h-4 text-text-secondary" />
        ) : (
          <ChevronUp className="w-4 h-4 text-text-secondary" />
        )}
      </button>

      {!collapsed && (
        <div className="px-4 pb-4 space-y-4 animate-fade-in">
          {/* Status Badge */}
          <div className="flex items-center gap-3">
            <div
              className={`flex items-center gap-2 px-3 py-1.5 rounded-full border ${
                isRunning
                  ? 'bg-success/10 border-success/30 text-success'
                  : 'bg-navy-700 border-navy-600 text-text-secondary'
              }`}
            >
              <span
                className={`w-2 h-2 rounded-full ${
                  isRunning ? 'bg-success animate-pulse-slow' : 'bg-text-secondary'
                }`}
              />
              <span className="text-xs font-medium">{isRunning ? 'Running' : 'Stopped'}</span>
            </div>
          </div>

          {/* Stats Grid */}
          <div className="grid grid-cols-2 gap-3">
            <div className="bg-navy-900/60 p-3 rounded">
              <div className="flex items-center gap-2 mb-1">
                <Zap className="w-3.5 h-3.5 text-accent" />
                <span className="text-xs text-text-secondary">Cycles</span>
              </div>
              <span className="text-xl font-mono font-semibold text-text-primary">
                {cycleCount.toLocaleString()}
              </span>
            </div>
            <div className="bg-navy-900/60 p-3 rounded">
              <div className="flex items-center gap-2 mb-1">
                <TrendingDown className="w-3.5 h-3.5 text-accent" />
                <span className="text-xs text-text-secondary">Best (nM)</span>
              </div>
              <span className="text-xl font-mono font-semibold text-accent">
                {bestAffinity < 0.001 ? '<0.001' : bestAffinity.toFixed(3)}
              </span>
            </div>
            <div className="bg-navy-900/60 p-3 rounded col-span-2">
              <div className="flex items-center gap-2 mb-1">
                <Target className="w-3.5 h-3.5 text-warning" />
                <span className="text-xs text-text-secondary">Target</span>
              </div>
              <span className="text-sm font-medium text-text-primary truncate block" title={targetName}>
                {targetName}
              </span>
            </div>
          </div>

          {/* Timing Info */}
          {status && (
            <div className="space-y-2">
              {status.last_cycle_time && (
                <div className="flex items-center gap-2 text-xs text-text-secondary">
                  <Clock className="w-3.5 h-3.5" />
                  <span>
                    Last cycle: {new Date(status.last_cycle_time).toLocaleTimeString()}
                  </span>
                </div>
              )}
              {status.next_cycle_time && isRunning && (
                <div className="flex items-center gap-2 text-xs text-text-secondary">
                  <Clock className="w-3.5 h-3.5" />
                  <span>
                    Next cycle: {new Date(status.next_cycle_time).toLocaleTimeString()}
                  </span>
                </div>
              )}
            </div>
          )}

          {/* Control Buttons */}
          <div className="flex items-center gap-2">
            {!isRunning ? (
              <button
                onClick={onStart}
                className="btn-primary flex-1 flex items-center justify-center gap-2"
              >
                <Play className="w-4 h-4" />
                Start
              </button>
            ) : (
              <button
                onClick={onStop}
                className="btn-secondary flex-1 flex items-center justify-center gap-2 border-danger/30 text-danger hover:bg-danger/10"
              >
                <Square className="w-4 h-4" />
                Stop
              </button>
            )}
            <button
              onClick={onStep}
              className="btn-secondary flex items-center justify-center gap-2"
              title="Run single cycle"
            >
              <StepForward className="w-4 h-4" />
              Step
            </button>
          </div>

          {/* Last Cycle Info */}
          {lastCycle && (
            <div className="bg-navy-900/60 p-3 rounded space-y-2">
              <div className="flex items-center gap-2">
                <Zap className="w-3.5 h-3.5 text-accent" />
                <span className="text-xs font-medium text-text-primary">
                  Cycle #{lastCycle.cycle_id}
                </span>
                {lastCycle.is_best_so_far && (
                  <span className="badge-cyan text-xs">NEW BEST</span>
                )}
              </div>
              <div className="flex items-center justify-between text-xs">
                <span className="text-text-secondary">Affinity</span>
                <span className="font-mono text-accent">
                  {lastCycle.predicted_affinity_nm.toFixed(3)} nM
                </span>
              </div>
              <div className="flex items-center justify-between text-xs">
                <span className="text-text-secondary">Improvement</span>
                <span
                  className={`font-mono ${
                    lastCycle.improvement > 0 ? 'text-success' : 'text-text-secondary'
                  }`}
                >
                  {lastCycle.improvement > 0 ? '+' : ''}
                  {lastCycle.improvement.toFixed(2)}x
                </span>
              </div>
            </div>
          )}

          {/* Recent Lessons */}
          <div>
            <div className="flex items-center gap-2 mb-2">
              <BookOpen className="w-3.5 h-3.5 text-text-secondary" />
              <span className="text-xs font-medium text-text-secondary">Recent Lessons</span>
            </div>
            <div className="bg-navy-900/60 rounded p-3 max-h-40 overflow-y-auto space-y-2">
              {lessons.map((lesson, i) => (
                <p key={i} className="text-xs text-text-primary leading-relaxed">
                  {lesson}
                </p>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
