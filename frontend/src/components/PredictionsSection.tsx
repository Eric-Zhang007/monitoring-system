import { useEffect, useMemo, useState } from 'react';
import { Brain, TrendingDown, TrendingUp } from 'lucide-react';

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';
const LIVE_SYMBOLS = ['BTC', 'ETH', 'SOL'];
const HORIZONS: Array<'1h' | '1d' | '7d'> = ['1h', '1d', '7d'];

type PredictRow = {
  horizon: '1h' | '1d' | '7d';
  direction: 'up' | 'down' | 'neutral';
  changePct: number;
  confidence: number;
  degraded: boolean;
  degradedReasons: string[];
};

const MOCK_PREDICTIONS: Record<string, PredictRow[]> = {
  BTC: [
    { horizon: '1h', direction: 'up', changePct: 1.2, confidence: 0.85, degraded: false, degradedReasons: [] },
    { horizon: '1d', direction: 'up', changePct: 3.5, confidence: 0.73, degraded: false, degradedReasons: [] },
    { horizon: '7d', direction: 'down', changePct: -2.1, confidence: 0.61, degraded: false, degradedReasons: [] },
  ],
  ETH: [
    { horizon: '1h', direction: 'up', changePct: 0.8, confidence: 0.79, degraded: false, degradedReasons: [] },
    { horizon: '1d', direction: 'neutral', changePct: 0.2, confidence: 0.65, degraded: false, degradedReasons: [] },
    { horizon: '7d', direction: 'down', changePct: -1.5, confidence: 0.58, degraded: false, degradedReasons: [] },
  ],
  SOL: [
    { horizon: '1h', direction: 'up', changePct: 0.6, confidence: 0.82, degraded: false, degradedReasons: [] },
    { horizon: '1d', direction: 'up', changePct: 2.1, confidence: 0.74, degraded: false, degradedReasons: [] },
    { horizon: '7d', direction: 'up', changePct: 4.2, confidence: 0.69, degraded: false, degradedReasons: [] },
  ],
};

interface PredictionsProps {
  symbol?: string;
}

function directionFromReturn(r: number): 'up' | 'down' | 'neutral' {
  if (r > 0.0005) return 'up';
  if (r < -0.0005) return 'down';
  return 'neutral';
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

export default function PredictionsSection({ symbol = 'BTC' }: PredictionsProps) {
  const normalizedSymbol = useMemo(() => {
    const s = String(symbol || 'BTC').toUpperCase();
    return LIVE_SYMBOLS.includes(s) ? s : 'BTC';
  }, [symbol]);

  const [rows, setRows] = useState<PredictRow[]>(MOCK_PREDICTIONS.BTC);
  const [mode, setMode] = useState<'live' | 'mock'>('mock');
  const [loading, setLoading] = useState<boolean>(false);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    const run = async () => {
      try {
        const out = await Promise.all(
          HORIZONS.map(async (h): Promise<PredictRow> => {
            const body = await postJson('/api/v2/predict/liquid', { symbol: normalizedSymbol, horizon: h });
            const expRet = Number(body?.expected_return ?? body?.outputs?.expected_return ?? 0);
            const conf = Number(body?.signal_confidence ?? body?.outputs?.signal_confidence ?? 0);
            return {
              horizon: h,
              direction: directionFromReturn(expRet),
              changePct: expRet * 100,
              confidence: Math.max(0, Math.min(1, conf)),
              degraded: Boolean(body?.degraded || body?.outputs?.degraded),
              degradedReasons: Array.isArray(body?.degraded_reasons) ? body.degraded_reasons : [],
            };
          })
        );
        if (cancelled) return;
        setRows(out);
        setMode('live');
      } catch {
        if (cancelled) return;
        setRows(MOCK_PREDICTIONS[normalizedSymbol] || MOCK_PREDICTIONS.BTC);
        setMode('mock');
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
  }, [normalizedSymbol]);

  const getDirectionClass = (direction: string, scheme: 'cn' | 'us' = 'cn') => {
    if (direction === 'up') {
      return scheme === 'cn'
        ? 'bg-red-100 text-red-700 dark:bg-red-900 dark:text-red-300'
        : 'bg-green-100 text-green-700 dark:bg-green-900 dark:text-green-300';
    }
    if (direction === 'down') {
      return scheme === 'cn'
        ? 'bg-green-100 text-green-700 dark:bg-green-900 dark:text-green-300'
        : 'bg-red-100 text-red-700 dark:bg-red-900 dark:text-red-300';
    }
    return 'bg-slate-100 text-slate-700 dark:bg-slate-700 dark:text-slate-300';
  };

  const getConfidenceLevel = (confidence: number) => {
    if (confidence >= 0.8) return { text: '高', color: 'bg-green-500' };
    if (confidence >= 0.65) return { text: '中', color: 'bg-yellow-500' };
    return { text: '低', color: 'bg-red-500' };
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center space-x-2 mb-6">
        <Brain className="w-6 h-6 text-purple-600 dark:text-purple-400" />
        <h2 className="text-2xl font-bold text-slate-900 dark:text-white">
          AI 预测分析 <span className="text-sm font-normal text-slate-500 dark:text-slate-400">| {normalizedSymbol}</span>
        </h2>
        <span
          className={`text-xs px-2 py-1 rounded ${mode === 'live' ? 'bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-300' : 'bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-300'}`}
        >
          {loading ? 'updating' : mode}
        </span>
      </div>

      <div className="bg-white dark:bg-slate-800 rounded-xl shadow-md p-6 border border-slate-200 dark:border-slate-700">
        <h3 className="text-lg font-semibold text-slate-900 dark:text-white mb-4">未来走势预测</h3>

        <div className="space-y-4">
          {rows.map((pred, index) => {
            const confidenceLevel = getConfidenceLevel(pred.confidence);
            return (
              <div
                key={`${pred.horizon}-${index}`}
                className="bg-slate-50 dark:bg-slate-700/50 rounded-lg p-4 border border-slate-200 dark:border-slate-600"
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-4">
                    <div className="w-16 text-center">
                      <div className="text-2xl font-bold text-slate-900 dark:text-white">{pred.horizon}</div>
                    </div>

                    <div className={`px-3 py-1.5 rounded-lg flex items-center space-x-2 ${getDirectionClass(pred.direction, 'cn')}`}>
                      {pred.direction === 'up' && <TrendingUp className="w-4 h-4" />}
                      {pred.direction === 'down' && <TrendingDown className="w-4 h-4" />}
                      {pred.direction === 'neutral' && <span className="font-semibold">•</span>}
                      <span className="font-semibold">
                        {pred.changePct >= 0 ? '+' : ''}
                        {pred.changePct.toFixed(2)}%
                      </span>
                    </div>
                  </div>

                  <div className="flex-1 max-w-xs mx-4">
                    <div className="flex items-center justify-between text-xs text-slate-500 dark:text-slate-400 mb-1">
                      <span>置信度</span>
                      <span>{(pred.confidence * 100).toFixed(0)}%</span>
                    </div>
                    <div className="h-2 bg-slate-200 dark:bg-slate-600 rounded-full overflow-hidden">
                      <div className={`h-full ${confidenceLevel.color} transition-all duration-300`} style={{ width: `${pred.confidence * 100}%` }} />
                    </div>
                  </div>

                  <div className="text-center min-w-20">
                    <div className="text-sm font-medium text-slate-700 dark:text-slate-300">{confidenceLevel.text}</div>
                    <div className="text-xs text-slate-500 dark:text-slate-400">置信等级</div>
                  </div>
                </div>
                {pred.degraded && pred.degradedReasons.length > 0 && (
                  <div className="mt-2 text-xs text-rose-600 dark:text-rose-300">
                    degraded: {pred.degradedReasons.join(', ')}
                  </div>
                )}
              </div>
            );
          })}
        </div>

        <div className="mt-6 p-4 bg-amber-50 dark:bg-amber-900/20 rounded-lg border border-amber-200 dark:border-amber-800">
          <p className="text-sm text-amber-800 dark:text-amber-300">
            免责声明：本预测仅供参考，不构成投资建议。投资有风险，需谨慎。
          </p>
        </div>
      </div>
    </div>
  );
}

