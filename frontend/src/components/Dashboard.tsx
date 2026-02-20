import { useEffect, useMemo, useState } from 'react';
import { ArrowDown, ArrowUp, TrendingUp, Wallet } from 'lucide-react';
import { useColorScheme } from '../contexts/ColorSchemeContext';

interface DashboardProps {
  activeSymbol: string;
  onSymbolChange: (symbol: string) => void;
}

type Snapshot = {
  price: number;
  changePct: number;
  volumeBillion: number;
  confidencePct: number;
  sentiment: number;
  volForecastPct: number;
  updatedAt: string;
  mode: 'live' | 'mock';
  degraded: boolean;
  degradedReasons: string[];
};

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';
const LIVE_SYMBOLS = ['BTC', 'ETH', 'SOL'];

const MOCK_PRICE_DATA: Record<string, Snapshot> = {
  BTC: {
    price: 67890.5,
    changePct: 2.35,
    volumeBillion: 28.5,
    confidencePct: 78.5,
    sentiment: 72,
    volForecastPct: 3.4,
    updatedAt: 'mock',
    mode: 'mock',
    degraded: false,
    degradedReasons: [],
  },
  ETH: {
    price: 3456.78,
    changePct: 1.89,
    volumeBillion: 15.3,
    confidencePct: 74.2,
    sentiment: 68,
    volForecastPct: 3.1,
    updatedAt: 'mock',
    mode: 'mock',
    degraded: false,
    degradedReasons: [],
  },
  SOL: {
    price: 168.23,
    changePct: 3.21,
    volumeBillion: 8.2,
    confidencePct: 76.8,
    sentiment: 70,
    volForecastPct: 4.6,
    updatedAt: 'mock',
    mode: 'mock',
    degraded: false,
    degradedReasons: [],
  },
};

async function fetchJson(path: string): Promise<any> {
  const res = await fetch(`${API_BASE}${path}`);
  const text = await res.text();
  const body = text ? JSON.parse(text) : {};
  if (!res.ok) {
    throw new Error(`${path} -> ${res.status}`);
  }
  return body;
}

async function postJson(path: string, payload: Record<string, any>): Promise<any> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(payload || {}),
  });
  const text = await res.text();
  const body = text ? JSON.parse(text) : {};
  if (!res.ok) {
    throw new Error(`${path} -> ${res.status}`);
  }
  return body;
}

function toLiveSnapshot(symbol: string, predictBody: any, featureBody: any): Snapshot {
  const outputs = (predictBody?.outputs || {}) as Record<string, any>;
  const featurePayload = (featureBody?.feature_payload || {}) as Record<string, any>;
  const expectedReturn = Number(predictBody?.expected_return ?? outputs?.expected_return ?? 0);
  const confidence = Number(predictBody?.signal_confidence ?? outputs?.signal_confidence ?? 0);
  const price = Number(outputs?.current_price ?? 0);
  const volForecast = Number(predictBody?.vol_forecast ?? outputs?.vol_forecast ?? 0);
  const logVolume = Number(featurePayload?.log_volume ?? 0);
  const estVolume = Math.max(0, Math.expm1(logVolume));
  const sentimentRaw =
    50 +
    25 * Number(featurePayload?.social_post_sentiment ?? 0) +
    25 * Number(featurePayload?.social_comment_sentiment ?? 0);

  return {
    price: Number.isFinite(price) && price > 0 ? price : Number(MOCK_PRICE_DATA[symbol]?.price || 0),
    changePct: Number.isFinite(expectedReturn) ? expectedReturn * 100 : 0,
    volumeBillion: estVolume > 0 ? estVolume / 1e8 : Number(MOCK_PRICE_DATA[symbol]?.volumeBillion || 0),
    confidencePct: Math.max(0, Math.min(100, confidence * 100)),
    sentiment: Math.max(0, Math.min(100, sentimentRaw)),
    volForecastPct: Math.max(0, volForecast * 100),
    updatedAt: String(featureBody?.feature_available_at || predictBody?.outputs?.as_of || new Date().toISOString()),
    mode: 'live',
    degraded: Boolean(predictBody?.degraded || outputs?.degraded),
    degradedReasons: Array.isArray(predictBody?.degraded_reasons) ? predictBody.degraded_reasons : [],
  };
}

export default function Dashboard({ activeSymbol, onSymbolChange }: DashboardProps) {
  const { scheme } = useColorScheme();
  const [snapshot, setSnapshot] = useState<Snapshot>(MOCK_PRICE_DATA.BTC);
  const [loading, setLoading] = useState<boolean>(false);

  useEffect(() => {
    if (!LIVE_SYMBOLS.includes(String(activeSymbol).toUpperCase())) {
      onSymbolChange('BTC');
    }
  }, [activeSymbol, onSymbolChange]);

  useEffect(() => {
    const symbol = String(activeSymbol || 'BTC').toUpperCase();
    let cancelled = false;
    setLoading(true);

    const run = async () => {
      try {
        const [predictBody, featureBody] = await Promise.all([
          postJson('/api/v2/predict/liquid', { symbol, horizon: '1d' }),
          fetchJson(`/api/v2/features/latest?target=${encodeURIComponent(symbol)}&track=liquid`),
        ]);
        if (cancelled) return;
        setSnapshot(toLiveSnapshot(symbol, predictBody, featureBody));
      } catch {
        if (cancelled) return;
        const fallback = MOCK_PRICE_DATA[symbol] || MOCK_PRICE_DATA.BTC;
        setSnapshot({ ...fallback, updatedAt: 'mock-fallback', mode: 'mock' });
      } finally {
        if (!cancelled) setLoading(false);
      }
    };

    run();
    const timer = setInterval(run, 15000);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, [activeSymbol]);

  const colorClass = useMemo(() => {
    if (scheme === 'cn') {
      return snapshot.changePct >= 0
        ? 'text-red-600 dark:text-red-400 bg-red-50 dark:bg-red-900/20'
        : 'text-green-600 dark:text-green-400 bg-green-50 dark:bg-green-900/20';
    }
    return snapshot.changePct >= 0
      ? 'text-green-600 dark:text-green-400 bg-green-50 dark:bg-green-900/20'
      : 'text-red-600 dark:text-red-400 bg-red-50 dark:bg-red-900/20';
  }, [scheme, snapshot.changePct]);

  return (
    <div className="space-y-6">
      <div className="bg-white dark:bg-slate-800 rounded-lg shadow-md p-4 border border-slate-200 dark:border-slate-700">
        <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
          选择标的
        </label>
        <select
          value={activeSymbol}
          onChange={(e) => onSymbolChange(e.target.value)}
          className="w-full px-4 py-2 rounded-md border border-slate-300 dark:border-slate-600 bg-slate-50 dark:bg-slate-700 text-slate-900 dark:text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent"
        >
          {LIVE_SYMBOLS.map((symbol) => (
            <option key={symbol} value={symbol}>
              {symbol}
            </option>
          ))}
        </select>
      </div>

      <div className="bg-white dark:bg-slate-800 rounded-xl shadow-lg p-6 border border-slate-200 dark:border-slate-700">
        <div className="flex items-start justify-between">
          <div>
            <h2 className="text-sm text-slate-500 dark:text-slate-400 font-medium mb-1">当前价格</h2>
            <div className="flex items-baseline space-x-3">
              <span className="text-4xl md:text-5xl font-bold text-slate-900 dark:text-white">
                $
                {snapshot.price.toLocaleString('en-US', {
                  minimumFractionDigits: 2,
                  maximumFractionDigits: 2,
                })}
              </span>
            </div>
            <div className="mt-2 text-xs">
              <span
                className={`inline-flex items-center px-2 py-1 rounded ${
                  snapshot.mode === 'live'
                    ? 'bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-300'
                    : 'bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-300'
                }`}
              >
                {snapshot.mode === 'live' ? 'live' : 'mock'}
              </span>
              {snapshot.degraded && (
                <span className="ml-2 inline-flex items-center px-2 py-1 rounded bg-rose-100 text-rose-700 dark:bg-rose-900/30 dark:text-rose-300">
                  degraded
                </span>
              )}
            </div>
          </div>
          <div className={`px-3 py-1.5 rounded-full flex items-center space-x-1 ${colorClass}`}>
            {snapshot.changePct >= 0 ? <ArrowUp className="w-4 h-4" /> : <ArrowDown className="w-4 h-4" />}
            <span className="text-sm font-semibold">
              {snapshot.changePct >= 0 ? '+' : ''}
              {snapshot.changePct.toFixed(2)}%
            </span>
          </div>
        </div>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-6 pt-6 border-t border-slate-200 dark:border-slate-700">
          <div>
            <p className="text-xs text-slate-500 dark:text-slate-400 mb-1">24H 成交量</p>
            <p className="text-lg font-semibold text-slate-900 dark:text-white">
              {snapshot.volumeBillion.toFixed(2)}亿
            </p>
          </div>
          <div>
            <p className="text-xs text-slate-500 dark:text-slate-400 mb-1">预测置信度</p>
            <p className="text-lg font-semibold text-slate-900 dark:text-white">{snapshot.confidencePct.toFixed(1)}%</p>
          </div>
          <div>
            <p className="text-xs text-slate-500 dark:text-slate-400 mb-1">情绪指数</p>
            <p className="text-lg font-semibold text-blue-600 dark:text-blue-400">{Math.round(snapshot.sentiment)}</p>
          </div>
          <div>
            <p className="text-xs text-slate-500 dark:text-slate-400 mb-1">更新时间</p>
            <p className="text-sm text-slate-600 dark:text-slate-300">
              {loading ? '刷新中' : String(snapshot.updatedAt || '').replace('T', ' ').slice(0, 19)}
            </p>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        <div className="bg-white dark:bg-slate-800 rounded-lg shadow-md p-5 border border-slate-200 dark:border-slate-700 hover:shadow-lg transition-shadow duration-200">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 bg-blue-100 dark:bg-blue-900 rounded-lg flex items-center justify-center">
              <TrendingUp className="w-5 h-5 text-blue-600 dark:text-blue-400" />
            </div>
            <div>
              <p className="text-xs text-slate-500 dark:text-slate-400">波动预测</p>
              <p className="text-lg font-semibold text-slate-900 dark:text-white">{snapshot.volForecastPct.toFixed(2)}%</p>
            </div>
          </div>
        </div>

        <div className="bg-white dark:bg-slate-800 rounded-lg shadow-md p-5 border border-slate-200 dark:border-slate-700 hover:shadow-lg transition-shadow duration-200">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 bg-orange-100 dark:bg-orange-900 rounded-lg flex items-center justify-center">
              <Wallet className="w-5 h-5 text-orange-600 dark:text-orange-400" />
            </div>
            <div>
              <p className="text-xs text-slate-500 dark:text-slate-400">建议</p>
              <p className="text-lg font-semibold text-slate-900 dark:text-white">
                {snapshot.changePct > 0.2 ? '偏多' : snapshot.changePct < -0.2 ? '偏空' : '观望'}
              </p>
            </div>
          </div>
        </div>

        <div className="bg-white dark:bg-slate-800 rounded-lg shadow-md p-5 border border-slate-200 dark:border-slate-700 hover:shadow-lg transition-shadow duration-200">
          <div className="text-xs text-slate-500 dark:text-slate-400">降级原因</div>
          <div className="mt-2 text-sm text-slate-800 dark:text-slate-200">
            {snapshot.degradedReasons.length ? snapshot.degradedReasons.join(', ') : 'none'}
          </div>
        </div>
      </div>
    </div>
  );
}

