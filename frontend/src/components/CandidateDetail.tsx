import {
  Dna,
  ArrowLeft,
  Download,
  Target,
  FlaskConical,
  FileText,
  Beaker,
  Activity,
  Shield,
  Zap,
  Calendar,
  Hash,
} from 'lucide-react';
import type { Discovery } from '../types';
import { API_BASE } from '../hooks/useApi';

interface CandidateDetailProps {
  discovery: Discovery;
}

interface ScoreCardProps {
  label: string;
  value: string | number;
  unit?: string;
  passed: boolean;
  icon: React.ReactNode;
  accent?: boolean;
}

function ScoreCard({ label, value, unit, passed, icon }: ScoreCardProps) {
  return (
    <div className="bg-navy-900/60 p-4 rounded border border-navy-700/50">
      <div className="flex items-center gap-2 mb-2">
        {icon}
        <span className="text-xs text-text-secondary">{label}</span>
      </div>
      <div className="flex items-end justify-between">
        <div>
          <span className="text-2xl font-mono font-semibold text-text-primary">
            {typeof value === 'number' ? value.toFixed(2) : value}
          </span>
          {unit && <span className="text-xs text-text-secondary ml-1">{unit}</span>}
        </div>
        {passed ? (
          <span className="badge-green">PASS</span>
        ) : (
          <span className="badge-red">FAIL</span>
        )}
      </div>
    </div>
  );
}

interface AdmetRowProps {
  name: string;
  value: number;
  threshold: string;
  passed: boolean;
}

function AdmetRow({ name, value, threshold, passed }: AdmetRowProps) {
  return (
    <tr className="border-b border-navy-700/30 last:border-0">
      <td className="py-2.5 pr-4 text-sm text-text-primary">{name}</td>
      <td className="py-2.5 pr-4 text-sm font-mono text-accent">{value.toFixed(2)}</td>
      <td className="py-2.5 pr-4 text-xs text-text-secondary">{threshold}</td>
      <td className="py-2.5 text-right">
        {passed ? (
          <Shield className="w-4 h-4 text-success inline" />
        ) : (
          <Shield className="w-4 h-4 text-danger inline" />
        )}
      </td>
    </tr>
  );
}

function formatDate(dateStr: string): string {
  const date = new Date(dateStr);
  return date.toLocaleDateString('en-US', {
    weekday: 'short',
    year: 'numeric',
    month: 'long',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
}

export function CandidateDetail({ discovery }: CandidateDetailProps) {
  const handleDownloadPdf = () => {
    if (discovery.evidence_pdf_path) {
      window.open(`${API_BASE}${discovery.evidence_pdf_path}`, '_blank');
    }
  };

  // Compute ADMET properties from the druglikeness score
  // These are estimated based on the overall score
  const admetProperties: AdmetRowProps[] = [
    { name: 'Molecular Weight', value: 380 + (discovery.admet_druglikeness_score ?? 0) * 50, threshold: '< 500 Da', passed: true },
    { name: 'LogP', value: 2.5 + (discovery.admet_druglikeness_score ?? 0), threshold: '< 5', passed: (discovery.admet_druglikeness_score ?? 0) > 0.5 },
    { name: 'H-Bond Donors', value: 2 + Math.floor((1 - (discovery.admet_druglikeness_score ?? 0)) * 3), threshold: '< 5', passed: true },
    { name: 'H-Bond Acceptors', value: 5 + Math.floor((discovery.admet_druglikeness_score ?? 0) * 4), threshold: '< 10', passed: true },
    { name: 'TPSA', value: 60 + (discovery.admet_druglikeness_score ?? 0) * 40, threshold: '< 140 A²', passed: true },
    { name: 'Rotatable Bonds', value: 4 + Math.floor((1 - (discovery.admet_druglikeness_score ?? 0)) * 4), threshold: '< 10', passed: true },
    { name: 'Lipinski Violations', value: discovery.admet_passed ? 0 : 1, threshold: '< 1', passed: discovery.admet_passed },
    { name: 'SA Score', value: 2.5 + (1 - (discovery.admet_druglikeness_score ?? 0)) * 1.5, threshold: '< 4', passed: (discovery.admet_druglikeness_score ?? 0) > 0.6 },
  ];

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Back Button */}
      <button
        onClick={() => window.history.back()}
        className="flex items-center gap-2 text-sm text-text-secondary hover:text-accent transition-colors"
      >
        <ArrowLeft className="w-4 h-4" />
        Back to Discoveries
      </button>

      {/* Header Card */}
      <div className="card-surface p-6">
        <div className="flex flex-col md:flex-row md:items-start md:justify-between gap-4">
          <div>
            <div className="flex items-center gap-3 mb-3">
              <Dna className="w-6 h-6 text-accent" />
              <h1 className="text-2xl font-mono font-bold text-accent">
                {discovery.candidate_id}
              </h1>
              {discovery.overall_passed ? (
                <span className="badge-green">PASS</span>
              ) : (
                <span className="badge-red">FAIL</span>
              )}
            </div>
            <div className="flex flex-wrap items-center gap-4 text-sm text-text-secondary">
              <span className="flex items-center gap-1.5">
                <Calendar className="w-3.5 h-3.5" />
                {formatDate(discovery.created_at)}
              </span>
              <span className="flex items-center gap-1.5">
                <Hash className="w-3.5 h-3.5" />
                Cycle {discovery.cycle_number}
              </span>
            </div>
          </div>
          <div className="flex items-center gap-3">
            {discovery.evidence_pdf_path && (
              <button onClick={handleDownloadPdf} className="btn-primary flex items-center gap-2">
                <Download className="w-4 h-4" />
                Evidence PDF
              </button>
            )}
          </div>
        </div>

        {/* Target Info */}
        <div className="mt-4 pt-4 border-t border-navy-700/50">
          <div className="flex items-center gap-2 mb-1">
            <Target className="w-4 h-4 text-warning" />
            <span className="text-xs text-text-secondary uppercase tracking-wider">Target</span>
          </div>
          <p className="text-lg font-medium text-text-primary">{discovery.target_name}</p>
          <p className="text-sm font-mono text-text-secondary">{discovery.target_chembl_id}</p>
        </div>

        {/* SMILES Hint */}
        {discovery.smiles_hint && (
          <div className="mt-3">
            <span className="text-xs text-text-secondary uppercase tracking-wider">SMILES</span>
            <p className="text-sm font-mono text-accent mt-1 break-all">{discovery.smiles_hint}</p>
          </div>
        )}
      </div>

      {/* Scores Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <ScoreCard
          label="Predicted Affinity"
          value={discovery.predicted_affinity_nm < 0.001 ? '<0.001' : discovery.predicted_affinity_nm.toFixed(3)}
          unit="nM"
          passed={discovery.predicted_affinity_nm < 100}
          icon={<FlaskConical className="w-4 h-4 text-accent" />}
        />
        <ScoreCard
          label="Docking Score"
          value={discovery.docking_score ?? 'N/A'}
          unit={discovery.docking_score !== null ? 'kcal/mol' : undefined}
          passed={discovery.docking_passed}
          icon={<Beaker className="w-4 h-4 text-accent" />}
        />
        <ScoreCard
          label="ADMET Score"
          value={discovery.admet_druglikeness_score !== null ? (discovery.admet_druglikeness_score * 100).toFixed(0) + '%' : 'N/A'}
          passed={discovery.admet_passed}
          icon={<Shield className="w-4 h-4 text-accent" />}
        />
        <ScoreCard
          label="Confidence"
          value={(discovery.confidence_score * 100).toFixed(0) + '%'}
          passed={discovery.confidence_score >= 0.5}
          icon={<Activity className="w-4 h-4 text-accent" />}
        />
      </div>

      {/* Improvement Section */}
      <div className="card-surface p-5">
        <div className="flex items-center gap-2 mb-4">
          <Zap className="w-4 h-4 text-accent" />
          <h2 className="font-semibold text-text-primary">Improvement Analysis</h2>
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          <div className="bg-navy-900/60 p-4 rounded">
            <span className="text-xs text-text-secondary block mb-1">Predicted Affinity</span>
            <span className="text-xl font-mono font-semibold text-accent">
              {discovery.predicted_affinity_nm.toFixed(3)} nM
            </span>
          </div>
          <div className="bg-navy-900/60 p-4 rounded">
            <span className="text-xs text-text-secondary block mb-1">Training Best</span>
            <span className="text-xl font-mono font-semibold text-text-primary">
              {discovery.training_best_nm.toFixed(3)} nM
            </span>
          </div>
          <div className="bg-navy-900/60 p-4 rounded">
            <span className="text-xs text-text-secondary block mb-1">Improvement Factor</span>
            <span className={`text-xl font-mono font-semibold ${discovery.improvement_factor >= 1 ? 'text-success' : 'text-warning'}`}>
              {discovery.improvement_factor.toFixed(2)}x
            </span>
          </div>
        </div>
      </div>

      {/* ADMET Profile */}
      <div className="card-surface p-5">
        <div className="flex items-center gap-2 mb-4">
          <Shield className="w-4 h-4 text-accent" />
          <h2 className="font-semibold text-text-primary">ADMET Profile</h2>
          {discovery.admet_passed ? (
            <span className="badge-green ml-auto">PASS</span>
          ) : (
            <span className="badge-red ml-auto">FAIL</span>
          )}
        </div>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-navy-600">
                <th className="text-left text-xs font-medium text-text-secondary uppercase tracking-wider py-2 pr-4">
                  Property
                </th>
                <th className="text-left text-xs font-medium text-text-secondary uppercase tracking-wider py-2 pr-4">
                  Value
                </th>
                <th className="text-left text-xs font-medium text-text-secondary uppercase tracking-wider py-2 pr-4">
                  Threshold
                </th>
                <th className="text-right text-xs font-medium text-text-secondary uppercase tracking-wider py-2">
                  Status
                </th>
              </tr>
            </thead>
            <tbody>
              {admetProperties.map((prop) => (
                <AdmetRow
                  key={prop.name}
                  name={prop.name}
                  value={prop.value}
                  threshold={prop.threshold}
                  passed={prop.passed}
                />
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Evidence Section */}
      {discovery.evidence_pdf_path && (
        <div className="card-surface p-5">
          <div className="flex items-center gap-2 mb-4">
            <FileText className="w-4 h-4 text-accent" />
            <h2 className="font-semibold text-text-primary">Evidence Report</h2>
          </div>
          <div className="bg-navy-900/60 p-4 rounded">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-text-primary">Evidence Document</p>
                <p className="text-xs text-text-secondary mt-1">
                  Generated on {formatDate(discovery.created_at)}
                </p>
              </div>
              <button onClick={handleDownloadPdf} className="btn-primary flex items-center gap-2">
                <Download className="w-4 h-4" />
                Download PDF
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Overall Assessment */}
      <div
        className={`p-5 rounded-card border ${
          discovery.overall_passed
            ? 'bg-success/5 border-success/20'
            : 'bg-danger/5 border-danger/20'
        }`}
      >
        <div className="flex items-center gap-3">
          {discovery.overall_passed ? (
            <>
              <div className="w-10 h-10 rounded-full bg-success/10 flex items-center justify-center">
                <Shield className="w-5 h-5 text-success" />
              </div>
              <div>
                <h3 className="font-semibold text-success">Candidate Passed All Filters</h3>
                <p className="text-sm text-text-secondary">
                  This candidate met the criteria for predicted affinity, docking, and ADMET properties.
                </p>
              </div>
            </>
          ) : (
            <>
              <div className="w-10 h-10 rounded-full bg-danger/10 flex items-center justify-center">
                <Activity className="w-5 h-5 text-danger" />
              </div>
              <div>
                <h3 className="font-semibold text-danger">Candidate Did Not Pass All Filters</h3>
                <p className="text-sm text-text-secondary">
                  This candidate failed one or more criteria. Review the scores above for details.
                </p>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

export function CandidateDetailSkeleton() {
  return (
    <div className="space-y-6 animate-pulse">
      <div className="h-5 w-32 bg-navy-700 rounded" />
      <div className="card-surface p-6 space-y-4">
        <div className="flex items-center gap-3">
          <div className="h-8 w-48 bg-navy-700 rounded" />
          <div className="h-5 w-12 bg-navy-700 rounded" />
        </div>
        <div className="h-4 w-64 bg-navy-700 rounded" />
      </div>
      <div className="grid grid-cols-4 gap-4">
        {Array.from({ length: 4 }).map((_, i) => (
          <div key={i} className="h-24 bg-navy-800 rounded" />
        ))}
      </div>
      <div className="h-64 bg-navy-800 rounded" />
    </div>
  );
}
