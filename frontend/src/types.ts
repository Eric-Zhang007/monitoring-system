// Types for the monitoring system
export interface PriceData {
  symbol: string;
  price: number;
  change: number;
  changePercent: number;
  volume: number;
  timestamp: string;
}

export interface NewsItem {
  id: number;
  title: string;
  symbol?: string;
  time: string;
  url: string;
  priority: 'high' | 'medium' | 'low';
  sentiment: 'positive' | 'negative' | 'neutral';
  is_important: boolean;
}

export interface Prediction {
  symbol: string;
  horizon: string;
  direction: 'up' | 'down' | 'neutral';
  change: string;
  confidence: 'high' | 'medium' | 'low';
  score: number;
  accuracy: number;
}

export interface SentimentAnalysis {
  symbol: string;
  sentiment: {
    positive: number;
    negative: number;
    neutral: number;
  };
  overall: 'positive' | 'negative' | 'neutral';
  trend: 'up' | 'down' | 'stable';
}

export interface SystemStatus {
  system: {
    available: boolean;
    total_gpus: number;
    gpu_memory: number;
    utilization: number;
  };
  services: {
    backend: string;
    redis_consumer: string;
    inference?: string;
    training?: string;
    collector?: string;
  };
}

export type ColorScheme = 'cn' | 'us'; // CN: 红涨绿跌, US: 绿涨红跌

export interface ColorSchemeContextType {
  scheme: ColorScheme;
  toggleScheme: () => void;
}

export interface WebSocketMessage {
  type: 'price_update' | 'news_update' | 'prediction' | 'pong' | 'signals' | 'events' | 'risk' | 'error';
  symbol?: string;
  price?: number;
  change?: string;
  status?: string;
  detail?: string;
  timestamp?: number | string;
  [key: string]: any;
}
