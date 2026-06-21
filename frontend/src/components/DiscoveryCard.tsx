import { Link } from 'react-router-dom';
import { ArrowRight, Beaker, Check, X, Dna } from 'lucide-react';
import type { Discovery } from '../types';

interface DiscoveryCardProps {
  discovery: Discovery;
}

function getAffinityBadgeClass(value: number): string {
  if (value < 1) return 'bg-accent/10 text-accent border-accent/20';
  if (value < 10) return 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20';
  if (value < 100) return 'bg-warning/10 text-warning border-warning/20';
  return 'bg-navy-700 text-text-secondary border-navy-600';
}

function getAffinityLabel(value: number): string {
  if (value < 1) return 'Exceptional';
  if (value < 10) return 'Strong';
  if (value < 100) return 'Moderate';
  return 'Weak';
}

function formatAffinity(value: number): string {
  if (value < 1) return value.toFixed(3);
  if (value < 10) return value.toFixed(2);
  if (value < 100) return value.toFixed(1);
  return value.toFixed(0);
}

function formatDate(dateStr: string): string {
  const date = new Date(dateStr);
  return date.toLocaleDateString('en-US', {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
  });
}

export function DiscoveryCard({ discovery }: DiscoveryCardProps) {
  const affinityClass = getAffinityBadgeClass(discovery.predicted_affinity_nm);
  const affinityLabel = getAffinityLabel(discovery.predicted_affinity_nm);

  return (
    <div className="card-surface p-5 transition-all duration-300 hover:-translate-y-1 hover:shadow-glow group animate-slide-up">
      {/* Header: Candidate ID + Date */}
      <div className="flex items-start justify-between mb-3">
        <div className="flex items-center gap-2 min-w-0">
          <Dna className="w-4 h-4 text-accent flex-shrink-0" />
          <h3 className="font-mono text-sm font-medium text-accent truncate">
            {discovery.candidate_id}
          </h3>
        </div>
        <span className="text-xs text-text-secondary flex-shrink-0 ml-2">
          {formatDate(discovery.created_at)}
        </span>
      </div>

      {/* Target Name */}
      <p className="text-sm text-text-primary mb-1 truncate" title={discovery.target_name}>
        {discovery.target_name}
      </p>
      <p className="text-xs text-text-secondary mb-4 font-mono">
        {discovery.target_chembl_id}
      </p>

      {/* Affinity Badge */}
      <div className={`inline-flex items-center gap-2 px-3 py-1.5 rounded-badge border mb-4 ${affinityClass}`}>
        <Beaker className="w-3.5 h-3.5" />
        <span className="font-mono text-sm font-semibold">
          {formatAffinity(discovery.predicted_affinity_nm)} nM
        </span>
        <span className="text-xs opacity-75">({affinityLabel})</span>
      </div>

      {/* Scores Grid */}
      <div className="grid grid-cols-2 gap-2 mb-4">
        {/* Docking */}
        <div className="flex items-center gap-2 px-2.5 py-2 bg-navy-900/60 rounded">
          <span className="text-xs text-text-secondary">Docking</span>
          {discovery.docking_passed ? (
            <Check className="w-3.5 h-3.5 text-success ml-auto" />
          ) : (
            <X className="w-3.5 h-3.5 text-danger ml-auto" />
          )}
        </div>
        {/* ADMET */}
        <div className="flex items-center gap-2 px-2.5 py-2 bg-navy-900/60 rounded">
          <span className="text-xs text-text-secondary">ADMET</span>
          {discovery.admet_passed ? (
            <Check className="w-3.5 h-3.5 text-success ml-auto" />
          ) : (
            <X className="w-3.5 h-3.5 text-danger ml-auto" />
          )}
        </div>
      </div>

      {/* Confidence Bar */}
      <div className="mb-4">
        <div className="flex items-center justify-between mb-1">
          <span className="text-xs text-text-secondary">Confidence</span>
          <span className="text-xs font-mono text-text-primary">
            {(discovery.confidence_score * 100).toFixed(0)}%
          </span>
        </div>
        <div className="w-full h-1.5 bg-navy-700 rounded-full overflow-hidden">
          <div
            className={`h-full rounded-full transition-all duration-500 ${
              discovery.confidence_score >= 0.8
                ? 'bg-accent'
                : discovery.confidence_score >= 0.5
                  ? 'bg-warning'
                  : 'bg-danger'
            }`}
            style={{ width: `${discovery.confidence_score * 100}%` }}
          />
        </div>
      </div>

      {/* Overall Pass/Fail */}
      <div className="flex items-center gap-2 mb-4">
        <span className="text-xs text-text-secondary">Overall:</span>
        {discovery.overall_passed ? (
          <span className="badge-green text-xs">PASS</span>
        ) : (
          <span className="badge-red text-xs">FAIL</span>
        )}
      </div>

      {/* View Details Link */}
      <Link
        to={`/discovery/${discovery.id}`}
        className="flex items-center gap-2 text-sm text-text-secondary hover:text-accent transition-colors group/link"
      >
        <span>View Details</span>
        <ArrowRight className="w-4 h-4 group-hover/link:translate-x-1 transition-transform" />
      </Link>
    </div>
  );
}

export function DiscoveryCardSkeleton() {
  return (
    <div className="card-surface p-5 animate-pulse">
      <div className="flex items-start justify-between mb-3">
        <div className="h-4 w-24 bg-navy-700 rounded" />
        <div className="h-3 w-16 bg-navy-700 rounded" />
      </div>
      <div className="h-4 w-full bg-navy-700 rounded mb-1" />
      <div className="h-3 w-20 bg-navy-700 rounded mb-4" />
      <div className="h-7 w-32 bg-navy-700 rounded mb-4" />
      <div className="grid grid-cols-2 gap-2 mb-4">
        <div className="h-8 bg-navy-700 rounded" />
        <div className="h-8 bg-navy-700 rounded" />
      </div>
      <div className="h-5 w-full bg-navy-700 rounded mb-4" />
      <div className="h-4 w-24 bg-navy-700 rounded" />
    </div>
  );
}
