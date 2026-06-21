import { useState, useEffect, useCallback, useRef } from 'react';

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

interface UseFetchResult<T> {
  data: T | null;
  loading: boolean;
  error: string | null;
  refetch: () => void;
}

export function useFetch<T>(url: string): UseFetchResult<T> {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const abortControllerRef = useRef<AbortController | null>(null);

  const fetchData = useCallback(async () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    abortControllerRef.current = new AbortController();

    setLoading(true);
    setError(null);

    try {
      const response = await fetch(`${API_BASE}${url}`, {
        signal: abortControllerRef.current.signal,
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        if (response.status === 404) {
          throw new Error('Resource not found');
        }
        if (response.status === 500) {
          throw new Error('Server error. Please try again later.');
        }
        throw new Error(`Request failed with status ${response.status}`);
      }

      const json = await response.json();
      setData(json);
    } catch (err) {
      if (err instanceof Error) {
        if (err.name === 'AbortError') {
          return;
        }
        setError(err.message);
      } else {
        setError('An unexpected error occurred');
      }
    } finally {
      setLoading(false);
    }
  }, [url]);

  useEffect(() => {
    fetchData();

    return () => {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
    };
  }, [fetchData]);

  return { data, loading, error, refetch: fetchData };
}

export function usePost<T, B = Record<string, unknown>>() {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const post = useCallback(async (url: string, body: B) => {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch(`${API_BASE}${url}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(body),
      });

      if (!response.ok) {
        throw new Error(`Request failed with status ${response.status}`);
      }

      const json = await response.json();
      setData(json);
      return json;
    } catch (err) {
      if (err instanceof Error) {
        setError(err.message);
      } else {
        setError('An unexpected error occurred');
      }
      return null;
    } finally {
      setLoading(false);
    }
  }, []);

  return { data, loading, error, post };
}

export { API_BASE };
