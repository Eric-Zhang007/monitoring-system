import { useEffect, useRef, useState } from 'react';

interface UseWebSocketReturn {
  isConnected: boolean;
  messages: any[];
  subscribe: (symbol: string) => void;
  unsubscribe: (symbol: string) => void;
}

const WS_URL = import.meta.env.VITE_WS_URL || 'ws://localhost:8000/stream/signals';

export function useWebSocket(): UseWebSocketReturn {
  const [isConnected, setIsConnected] = useState(false);
  const [messages, setMessages] = useState<any[]>([]);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    // Only connect in browser environment
    if (typeof window === 'undefined') {
      return;
    }

    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;

    ws.onopen = () => {
      setIsConnected(true);
      console.log('✅ WebSocket connected');
    };

    ws.onclose = () => {
      setIsConnected(false);
      console.log('❌ WebSocket disconnected');
    };

    ws.onerror = (error) => {
      console.error('❌ WebSocket error:', error);
    };

    ws.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data);
        setMessages(prev => [...prev, message].slice(-100)); // Keep last 100 messages
      } catch (e) {
        console.error('❌ Failed to parse WebSocket message:', e);
      }
    };

    // Cleanup on unmount
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);

  const subscribe = (symbol: string) => {
    if (wsRef.current && isConnected) {
      wsRef.current.send(JSON.stringify({
        type: 'subscribe',
        channel: 'signals',
        symbol: symbol,
      }));
    }
  };

  const unsubscribe = (symbol: string) => {
    if (wsRef.current && isConnected) {
      wsRef.current.send(JSON.stringify({
        type: 'unsubscribe',
        channel: 'signals',
        symbol: symbol,
      }));
    }
  };

  return {
    isConnected,
    messages,
    subscribe,
    unsubscribe,
  };
}
