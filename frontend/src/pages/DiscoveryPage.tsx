import { useParams, Link } from 'react-router-dom';
import { ArrowLeft, AlertTriangle } from 'lucide-react';
import { Header } from '../components/Header';
import { CandidateDetail, CandidateDetailSkeleton } from '../components/CandidateDetail';
import { useFetch } from '../hooks/useApi';
import type { Discovery } from '../types';

export function DiscoveryPage() {
  const { id } = useParams<{ id: string }>();
  const { data: discovery, loading, error } = useFetch<Discovery>(`/api/discoveries/${id}`);

  return (
    <div className="min-h-screen bg-navy-900">
      <Header />

      <main className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Breadcrumb */}
        <nav className="mb-6">
          <Link
            to="/"
            className="inline-flex items-center gap-2 text-sm text-text-secondary hover:text-accent transition-colors"
          >
            <ArrowLeft className="w-4 h-4" />
            Back to Discoveries
          </Link>
        </nav>

        {/* Content States */}
        {loading ? (
          <CandidateDetailSkeleton />
        ) : error ? (
          <div className="card-surface p-12 text-center animate-fade-in">
            <AlertTriangle className="w-12 h-12 text-warning mx-auto mb-4" />
            <h2 className="text-lg font-medium text-text-primary mb-2">
              Failed to Load Discovery
            </h2>
            <p className="text-sm text-text-secondary mb-4">{error}</p>
            <Link to="/" className="btn-primary">
              Return to Discoveries
            </Link>
          </div>
        ) : !discovery ? (
          <div className="card-surface p-12 text-center animate-fade-in">
            <AlertTriangle className="w-12 h-12 text-warning mx-auto mb-4" />
            <h2 className="text-lg font-medium text-text-primary mb-2">
              Discovery Not Found
            </h2>
            <p className="text-sm text-text-secondary mb-4">
              The discovery with ID <span className="font-mono text-accent">{id}</span> could not be found.
            </p>
            <Link to="/" className="btn-primary">
              Return to Discoveries
            </Link>
          </div>
        ) : (
          <CandidateDetail discovery={discovery} />
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-navy-700/50 mt-12">
        <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 py-6 text-center text-xs text-text-secondary">
          ASI-Evolve Discovery Engine &copy; {new Date().getFullYear()}
        </div>
      </footer>
    </div>
  );
}
