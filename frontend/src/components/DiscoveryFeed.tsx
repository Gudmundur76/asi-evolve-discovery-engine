import { useState, useMemo } from 'react';
import { Filter, SortDesc, Search, FlaskConical, SlidersHorizontal } from 'lucide-react';
import { useFetch } from '../hooks/useApi';
import { DiscoveryCard, DiscoveryCardSkeleton } from './DiscoveryCard';
import type { Discovery, DiscoveryFilters } from '../types';

const ITEMS_PER_PAGE = 12;

export function DiscoveryFeed() {
  const { data, loading, error, refetch } = useFetch<Discovery[]>('/api/discoveries');
  const [filters, setFilters] = useState<DiscoveryFilters>({
    dateFrom: '',
    dateTo: '',
    minConfidence: 0,
    sortBy: 'date',
    sortOrder: 'desc',
  });
  const [page, setPage] = useState(1);
  const [showFilters, setShowFilters] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');

  const filteredAndSorted = useMemo(() => {
    if (!data) return [];

    let result = [...data];

    // Search filter
    if (searchQuery.trim()) {
      const q = searchQuery.toLowerCase();
      result = result.filter(
        (d) =>
          d.candidate_id.toLowerCase().includes(q) ||
          d.target_name.toLowerCase().includes(q) ||
          d.target_chembl_id.toLowerCase().includes(q)
      );
    }

    // Date range
    if (filters.dateFrom) {
      const from = new Date(filters.dateFrom).getTime();
      result = result.filter((d) => new Date(d.created_at).getTime() >= from);
    }
    if (filters.dateTo) {
      const to = new Date(filters.dateTo).getTime();
      result = result.filter((d) => new Date(d.created_at).getTime() <= to);
    }

    // Confidence
    if (filters.minConfidence > 0) {
      result = result.filter((d) => d.confidence_score >= filters.minConfidence);
    }

    // Sort
    result.sort((a, b) => {
      const order = filters.sortOrder === 'asc' ? 1 : -1;
      switch (filters.sortBy) {
        case 'affinity':
          return (a.predicted_affinity_nm - b.predicted_affinity_nm) * order;
        case 'confidence':
          return (a.confidence_score - b.confidence_score) * order;
        case 'date':
        default:
          return (new Date(a.created_at).getTime() - new Date(b.created_at).getTime()) * order;
      }
    });

    return result;
  }, [data, filters, searchQuery]);

  const paginated = useMemo(() => {
    const start = 0;
    const end = page * ITEMS_PER_PAGE;
    return filteredAndSorted.slice(start, end);
  }, [filteredAndSorted, page]);

  const hasMore = paginated.length < filteredAndSorted.length;

  const handleSortChange = (sortBy: DiscoveryFilters['sortBy']) => {
    setFilters((prev) => ({
      ...prev,
      sortBy,
      sortOrder: prev.sortBy === sortBy && prev.sortOrder === 'desc' ? 'asc' : 'desc',
    }));
    setPage(1);
  };

  if (error) {
    return (
      <div className="card-surface p-8 text-center">
        <p className="text-danger mb-4">{error}</p>
        <button onClick={refetch} className="btn-primary">
          Retry
        </button>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Filter Bar */}
      <div className="card-surface p-4 space-y-4">
        {/* Search Row */}
        <div className="flex flex-col sm:flex-row gap-3">
          <div className="relative flex-1">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-text-secondary" />
            <input
              type="text"
              placeholder="Search candidates, targets..."
              value={searchQuery}
              onChange={(e) => {
                setSearchQuery(e.target.value);
                setPage(1);
              }}
              className="w-full pl-10 pr-4 py-2.5 bg-navy-900 border border-navy-700 rounded-card text-sm text-text-primary placeholder-text-secondary/60 focus:outline-none focus:border-accent/50 transition-colors"
            />
          </div>
          <button
            onClick={() => setShowFilters(!showFilters)}
            className={`btn-secondary flex items-center gap-2 ${showFilters ? 'text-accent border-accent/30' : ''}`}
          >
            <SlidersHorizontal className="w-4 h-4" />
            <span className="hidden sm:inline">Filters</span>
          </button>
        </div>

        {/* Expandable Filters */}
        {showFilters && (
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 pt-3 border-t border-navy-700/50 animate-fade-in">
            <div>
              <label className="block text-xs text-text-secondary mb-1">Date From</label>
              <input
                type="date"
                value={filters.dateFrom}
                onChange={(e) => {
                  setFilters((prev) => ({ ...prev, dateFrom: e.target.value }));
                  setPage(1);
                }}
                className="w-full px-3 py-2 bg-navy-900 border border-navy-700 rounded-card text-sm text-text-primary focus:outline-none focus:border-accent/50"
              />
            </div>
            <div>
              <label className="block text-xs text-text-secondary mb-1">Date To</label>
              <input
                type="date"
                value={filters.dateTo}
                onChange={(e) => {
                  setFilters((prev) => ({ ...prev, dateTo: e.target.value }));
                  setPage(1);
                }}
                className="w-full px-3 py-2 bg-navy-900 border border-navy-700 rounded-card text-sm text-text-primary focus:outline-none focus:border-accent/50"
              />
            </div>
            <div>
              <label className="block text-xs text-text-secondary mb-1">
                Min Confidence: {(filters.minConfidence * 100).toFixed(0)}%
              </label>
              <input
                type="range"
                min="0"
                max="1"
                step="0.05"
                value={filters.minConfidence}
                onChange={(e) => {
                  setFilters((prev) => ({
                    ...prev,
                    minConfidence: parseFloat(e.target.value),
                  }));
                  setPage(1);
                }}
                className="w-full h-2 bg-navy-700 rounded-full appearance-none cursor-pointer accent-accent mt-2"
              />
            </div>
          </div>
        )}

        {/* Sort + Count */}
        <div className="flex items-center justify-between pt-2 border-t border-navy-700/50">
          <div className="flex items-center gap-2 text-xs text-text-secondary">
            <Filter className="w-3.5 h-3.5" />
            <span>
              {filteredAndSorted.length} result{filteredAndSorted.length !== 1 ? 's' : ''}
            </span>
          </div>
          <div className="flex items-center gap-1">
            {(['date', 'affinity', 'confidence'] as const).map((key) => (
              <button
                key={key}
                onClick={() => handleSortChange(key)}
                className={`px-3 py-1.5 text-xs font-medium rounded-badge border transition-all capitalize ${
                  filters.sortBy === key
                    ? 'text-accent border-accent/30 bg-accent/10'
                    : 'text-text-secondary border-navy-700 hover:text-text-primary hover:border-navy-600'
                }`}
              >
                <SortDesc className="w-3 h-3 inline mr-1" />
                {key}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Discovery Grid */}
      {loading && !data ? (
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
          {Array.from({ length: 6 }).map((_, i) => (
            <DiscoveryCardSkeleton key={i} />
          ))}
        </div>
      ) : paginated.length === 0 ? (
        <div className="card-surface p-12 text-center">
          <FlaskConical className="w-12 h-12 text-text-secondary mx-auto mb-4" />
          <h3 className="text-lg font-medium text-text-primary mb-2">No discoveries found</h3>
          <p className="text-sm text-text-secondary mb-4">
            {searchQuery || filters.minConfidence > 0
              ? 'Try adjusting your filters to see more results.'
              : 'The discovery loop hasn\'t produced any results yet. Start the loop to begin generating candidates.'}
          </p>
          {(searchQuery || filters.minConfidence > 0) && (
            <button
              onClick={() => {
                setSearchQuery('');
                setFilters((prev) => ({ ...prev, minConfidence: 0, dateFrom: '', dateTo: '' }));
              }}
              className="btn-primary"
            >
              Clear Filters
            </button>
          )}
        </div>
      ) : (
        <>
          <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
            {paginated.map((discovery) => (
              <DiscoveryCard key={discovery.id} discovery={discovery} />
            ))}
          </div>

          {hasMore && (
            <div className="flex justify-center pt-4">
              <button onClick={() => setPage((p) => p + 1)} className="btn-secondary">
                Load More ({filteredAndSorted.length - paginated.length} remaining)
              </button>
            </div>
          )}
        </>
      )}
    </div>
  );
}
