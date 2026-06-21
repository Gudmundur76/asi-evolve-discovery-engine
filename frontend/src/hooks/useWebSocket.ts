import { useState, useEffect, useRef, useCallback } from 'react';
import type { LoopStatus, Discovery, CycleResult } from '../types';

interface WebSocketState {
  status: LoopStatus | null;
  lastCycle: CycleResult | null;
  discoveries: Discovery[];
  connected: boolean;
  connectionError: string | null;
}

const WS_URL = 'ws://localhost:8000/ws';
const INITIAL_RECONNECT_DELAY = 1000;
const MAX_RECONNECT_DELAY = 30000;

export function useWebSocket(): WebSocketState & {
  sendMessage: (msg: Record<string, unknown>) => void;
} {
  const [status, setStatus] = useState<LoopStatus | null>(null);
  const [lastCycle, setLastCycle] = useState<CycleResult | null>(null);
  const [discoveries, setDiscoveries] = useState<Discovery[]>([]);
  const [connected, setConnected] = useState(false);
  const [connectionError, setConnectionError] = useState<string | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectDelayRef = useRef(INITIAL_RECONNECT_DELAY);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const isUnmountingRef = useRef(false);

  const connect = useCallback(() => {
    if (isUnmountingRef.current) return;

    try {
      const ws = new WebSocket(WS_URL);
      wsRef.current = ws;

      ws.onopen = () => {
        if (isUnmountingRef.current) return;
        setConnected(true);
        setConnectionError(null);
        reconnectDelayRef.current = INITIAL_RECONNECT_DELAY;
      };

      ws.onmessage = (event) => {
        if (isUnmountingRef.current) return;

        try {
          const message = JSON.parse(event.data);

          switch (message.type) {
            case 'loop_status':
              setStatus(message.payload as LoopStatus);
              break;
            case 'new_discovery':
              setDiscoveries((prev) => {
                const newDiscovery = message.payload as Discovery;
                if (prev.some((d) => d.id === newDiscovery.id)) {
                  return prev;
                }
                return [newDiscovery, ...prev];
              });
              break;
            case 'cycle_complete':
              setLastCycle(message.payload as CycleResult);
              break;
            case 'error':
              setConnectionError(
                (message.payload as { message: string }).message || 'WebSocket error'
              );
              break;
            default:
              break;
          }
        } catch {
          // Ignore malformed messages
        }
      };

      ws.onclose = () => {
        if (isUnmountingRef.current) return;
        setConnected(false);

        // Exponential backoff
        reconnectDelayRef.current = Math.min(
          reconnectDelayRef.current * 2,
          MAX_RECONNECT_DELAY
        );

        reconnectTimerRef.current = setTimeout(() => {
          connect();
        }, reconnectDelayRef.current);
      };

      ws.onerror = () => {
        setConnectionError('WebSocket connection failed. Retrying...');
        ws.close();
      };
    } catch {
      setConnectionError('Failed to create WebSocket connection');
    }
  }, []);

  const sendMessage = useCallback((msg: Record<string, unknown>) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(msg));
    }
  }, []);

  useEffect(() => {
    isUnmountingRef.current = false;
    connect();

    return () => {
      isUnmountingRef.current = true;
      if (reconnectTimerRef.current) {
        clearTimeout(reconnectTimerRef.current);
      }
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [connect]);

  return {
    status,
    lastCycle,
    discoveries,
    connected,
    connectionError,
    sendMessage,
  };
}
