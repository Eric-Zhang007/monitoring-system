import { useCallback, useEffect, useMemo, useState } from 'react';
import { Activity, AlertTriangle, CheckCircle, Cpu, Database, RefreshCw, Server } from 'lucide-react';

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

type Dict = Record<string, any>;
type LogLevel = 'info' | 'warning' | 'error' | 'success';

interface LogItem {
  id: number;
  level: LogLevel;
  message: string;
  timestamp: string;
}

function nowLabel(): string {
  const d = new Date();
  return d.toISOString().replace('T', ' ').slice(0, 19);
}

function num(v: any, fallback = 0): number {
  const n = Number(v);
  return Number.isFinite(n) ? n : fallback;
}

async function fetchJson(path: string): Promise<Dict> {
  const res = await fetch(`${API_BASE}${path}`);
  const text = await res.text();
  let body: Dict = {};
  try {
    body = text ? JSON.parse(text) : {};
  } catch {
    body = { raw: text };
  }
  if (!res.ok) {
    throw new Error(`${path} -> ${res.status}`);
  }
  return body;
}

async function postJson(path: string, payload: Dict): Promise<Dict> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(payload || {}),
  });
  const text = await res.text();
  let body: Dict = {};
  try {
    body = text ? JSON.parse(text) : {};
  } catch {
    body = { raw: text };
  }
  if (!res.ok) {
    throw new Error(`${path} -> ${res.status}`);
  }
  return body;
}

export default function MonitorPanel() {
  const [status, setStatus] = useState<Dict>({});
  const [health, setHealth] = useState<Dict>({});
  const [rollout, setRollout] = useState<Dict>({});
  const [killSwitch, setKillSwitch] = useState<Dict>({});
  const [liveControl, setLiveControl] = useState<Dict>({});
  const [pnl, setPnl] = useState<Dict>({});
  const [opsState, setOpsState] = useState<Dict>({});
  const [historyCompleteness, setHistoryCompleteness] = useState<Dict>({});
  const [alignment, setAlignment] = useState<Dict>({});
  const [socialThroughput, setSocialThroughput] = useState<Dict>({});
  const [socialCoverage, setSocialCoverage] = useState<Dict>({});
  const [modelStatus, setModelStatus] = useState<Dict>({});
  const [paperPerf, setPaperPerf] = useState<Dict>({});
  const [controlBusy, setControlBusy] = useState<boolean>(false);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [logs, setLogs] = useState<LogItem[]>([]);

  const appendLog = useCallback((level: LogLevel, message: string) => {
    setLogs((prev) => [
      { id: Date.now() + Math.floor(Math.random() * 1000), level, message, timestamp: nowLabel() },
      ...prev,
    ].slice(0, 120));
  }, []);

  const load = useCallback(async () => {
    const start = Date.now();
    try {
      const [s, h, r, k, lc, p, o, hc, al, st, sc, ms, pp] = await Promise.all([
        fetchJson('/api/status'),
        fetchJson('/health'),
        fetchJson('/api/v2/models/rollout/state?track=liquid'),
        fetchJson('/api/v2/risk/kill-switch?track=liquid&strategy_id=global'),
        fetchJson('/api/v2/control/live/state'),
        fetchJson('/api/v2/metrics/pnl-attribution?track=liquid&lookback_hours=168'),
        fetchJson('/api/ops/runtime-state'),
        fetchJson('/api/v2/monitor/history-completeness'),
        fetchJson('/api/v2/monitor/alignment'),
        fetchJson('/api/v2/monitor/social-throughput'),
        fetchJson('/api/v2/dq/social-coverage?window_hours=24'),
        fetchJson('/api/v2/monitor/model-status'),
        fetchJson('/api/v2/monitor/paper-performance'),
      ]);
      setStatus(s || {});
      setHealth(h || {});
      setRollout(r || {});
      setKillSwitch(k || {});
      setLiveControl(lc || {});
      setPnl(p || {});
      setOpsState(o || {});
      setHistoryCompleteness(hc || {});
      setAlignment(al || {});
      setSocialThroughput(st || {});
      setSocialCoverage(sc || {});
      setModelStatus(ms || {});
      setPaperPerf(pp || {});
      const warn = Array.isArray(s?.warnings) ? s.warnings : [];
      if (warn.length > 0) {
        appendLog('warning', `status warnings: ${warn.slice(0, 2).join(' | ')}`);
      } else {
        appendLog('success', `monitor refresh ok (${Date.now() - start}ms)`);
      }
    } catch (err: any) {
      appendLog('error', `monitor refresh failed: ${String(err?.message || err)}`);
    } finally {
      setIsLoading(false);
    }
  }, [appendLog]);

  const sendControl = useCallback(async (path: string, payload: Dict, okMsg: string) => {
    try {
      setControlBusy(true);
      await postJson(path, payload);
      appendLog('success', okMsg);
      await load();
    } catch (err: any) {
      appendLog('error', `control failed: ${String(err?.message || err)}`);
    } finally {
      setControlBusy(false);
    }
  }, [appendLog, load]);

  useEffect(() => {
    load();
    const timer = setInterval(load, 15000);
    return () => clearInterval(timer);
  }, [load]);

  const gpuRows = useMemo(() => {
    const rows = status?.system?.gpu?.gpus;
    return Array.isArray(rows) ? rows : [];
  }, [status]);

  const avgGpuUtil = useMemo(() => {
    if (!gpuRows.length) return 0;
    return gpuRows.reduce((acc: number, x: Dict) => acc + num(x?.usage_percent), 0) / gpuRows.length;
  }, [gpuRows]);

  const totalGpuMem = useMemo(
    () => gpuRows.reduce((acc: number, x: Dict) => acc + num(x?.total_gb), 0),
    [gpuRows],
  );

  const services = useMemo(() => {
    const map = status?.services;
    return map && typeof map === 'object' ? Object.entries(map) : [];
  }, [status]);

  const systemHealthy = String(health?.status || '').toLowerCase() === 'healthy';
  const killTriggered = Boolean(killSwitch?.triggered);
  const dataStats = status?.data || {};
  const pnlTotals = pnl?.totals || {};
  const opsCycle = num(opsState?.cycle, 0);
  const liveEnabled = Boolean(liveControl?.live_enabled ?? paperPerf?.live_enabled);
  const paperEnabled = Boolean(liveControl?.paper_enabled ?? paperPerf?.paper_enabled);
  const latestCandidate = modelStatus?.latest_candidate || {};

  const getLogIcon = (level: LogLevel) => {
    if (level === 'error') return <AlertTriangle className="w-4 h-4 text-red-500" />;
    if (level === 'warning') return <AlertTriangle className="w-4 h-4 text-yellow-500" />;
    if (level === 'success') return <CheckCircle className="w-4 h-4 text-green-500" />;
    return <Activity className="w-4 h-4 text-blue-500" />;
  };

  const getLogColor = (level: LogLevel) => {
    if (level === 'error') return 'text-red-600 dark:text-red-400 bg-red-50 dark:bg-red-900/20';
    if (level === 'warning') return 'text-yellow-600 dark:text-yellow-400 bg-yellow-50 dark:bg-yellow-900/20';
    if (level === 'success') return 'text-green-600 dark:text-green-400 bg-green-50 dark:bg-green-900/20';
    return 'text-blue-600 dark:text-blue-400 bg-blue-50 dark:bg-blue-900/20';
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center space-x-2">
          <Server className="w-6 h-6 text-orange-600 dark:text-orange-400" />
          <h2 className="text-2xl font-bold text-slate-900 dark:text-white">系统监控</h2>
        </div>
        <button
          onClick={load}
          className="inline-flex items-center gap-2 px-3 py-2 rounded-lg bg-slate-100 dark:bg-slate-700 text-slate-700 dark:text-slate-200 hover:opacity-90"
        >
          <RefreshCw className="w-4 h-4" />
          刷新
        </button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="bg-white dark:bg-slate-800 rounded-lg shadow-md p-5 border border-slate-200 dark:border-slate-700">
          <div className="flex items-center space-x-3 mb-3">
            <CheckCircle className={`w-5 h-5 ${systemHealthy ? 'text-green-500' : 'text-red-500'}`} />
            <h3 className="text-sm font-medium text-slate-700 dark:text-slate-300">系统健康</h3>
          </div>
          <p className={`text-2xl font-bold ${systemHealthy ? 'text-green-600' : 'text-red-600'}`}>
            {systemHealthy ? 'HEALTHY' : 'DEGRADED'}
          </p>
          <p className="text-xs text-slate-500 dark:text-slate-400">health / status</p>
        </div>

        <div className="bg-white dark:bg-slate-800 rounded-lg shadow-md p-5 border border-slate-200 dark:border-slate-700">
          <div className="flex items-center space-x-3 mb-3">
            <Cpu className="w-5 h-5 text-blue-600 dark:text-blue-400" />
            <h3 className="text-sm font-medium text-slate-700 dark:text-slate-300">GPU 利用率</h3>
          </div>
          <p className="text-2xl font-bold text-slate-900 dark:text-white">{avgGpuUtil.toFixed(1)}%</p>
          <p className="text-xs text-slate-500 dark:text-slate-400">
            {gpuRows.length} GPUs / {totalGpuMem.toFixed(0)} GB
          </p>
          <div className="mt-3 h-2 bg-slate-200 dark:bg-slate-600 rounded-full overflow-hidden">
            <div className="h-full bg-blue-600 dark:bg-blue-400" style={{ width: `${Math.min(100, avgGpuUtil)}%` }} />
          </div>
        </div>

        <div className="bg-white dark:bg-slate-800 rounded-lg shadow-md p-5 border border-slate-200 dark:border-slate-700">
          <div className="flex items-center space-x-3 mb-3">
            <Database className="w-5 h-5 text-orange-600 dark:text-orange-400" />
            <h3 className="text-sm font-medium text-slate-700 dark:text-slate-300">数据活跃度</h3>
          </div>
          <p className="text-2xl font-bold text-slate-900 dark:text-white">{num(dataStats?.active_symbols, 0)}</p>
          <p className="text-xs text-slate-500 dark:text-slate-400">
            news24h={num(dataStats?.recent_news_24h, 0)} pred1h={num(dataStats?.recent_predictions_1h, 0)}
          </p>
        </div>

        <div className="bg-white dark:bg-slate-800 rounded-lg shadow-md p-5 border border-slate-200 dark:border-slate-700">
          <div className="flex items-center space-x-3 mb-3">
            <AlertTriangle className={`w-5 h-5 ${killTriggered ? 'text-red-500' : 'text-green-500'}`} />
            <h3 className="text-sm font-medium text-slate-700 dark:text-slate-300">风控开关</h3>
          </div>
          <p className={`text-2xl font-bold ${killTriggered ? 'text-red-600' : 'text-green-600'}`}>
            {killTriggered ? 'TRIGGERED' : 'ARMED'}
          </p>
          <p className="text-xs text-slate-500 dark:text-slate-400">
            remaining={num(killSwitch?.remaining_seconds, 0)}s
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <div className="bg-white dark:bg-slate-800 rounded-xl shadow-md p-5 border border-slate-200 dark:border-slate-700">
          <h3 className="text-lg font-semibold text-slate-900 dark:text-white mb-3">全历史完整性</h3>
          <div className="text-sm text-slate-700 dark:text-slate-300 space-y-1">
            <div>complete: {String(historyCompleteness?.summary?.history_window_complete ?? 'n/a')}</div>
            <div>comment_ratio_ge_10x: {String(historyCompleteness?.summary?.comment_ratio_ge_10x ?? 'n/a')}</div>
            <div>ratio: {num(historyCompleteness?.summary?.full_window_ratio, 0).toFixed(4)}</div>
          </div>
        </div>

        <div className="bg-white dark:bg-slate-800 rounded-xl shadow-md p-5 border border-slate-200 dark:border-slate-700">
          <h3 className="text-lg font-semibold text-slate-900 dark:text-white mb-3">As-Of 对齐</h3>
          <div className="text-sm text-slate-700 dark:text-slate-300 space-y-1">
            <div>future_leakage: {num(alignment?.future_leakage_count, 0)}</div>
            <div>match_rate: {(num(alignment?.snapshot_sample_time_match_rate, 0) * 100).toFixed(2)}%</div>
            <div>checked_rows: {num(alignment?.checked_rows, 0)}</div>
          </div>
        </div>

        <div className="bg-white dark:bg-slate-800 rounded-xl shadow-md p-5 border border-slate-200 dark:border-slate-700">
          <h3 className="text-lg font-semibold text-slate-900 dark:text-white mb-3">社媒吞吐</h3>
          <div className="text-sm text-slate-700 dark:text-slate-300 space-y-1">
            <div>posts_added: {num(socialThroughput?.posts_added, 0)}</div>
            <div>comments_added: {num(socialThroughput?.comments_added, 0)}</div>
            <div>new_ratio: {num(socialThroughput?.new_ratio, 0).toFixed(4)}</div>
            <div>coverage_ratio: {num(socialCoverage?.totals?.coverage_ratio, 0).toFixed(4)}</div>
            <div>lag_p90_sec: {num(socialCoverage?.totals?.ingest_lag_p90_sec, 0).toFixed(1)}</div>
            <div>lag_p99_sec: {num(socialCoverage?.totals?.ingest_lag_p99_sec, 0).toFixed(1)}</div>
          </div>
        </div>

        <div className="bg-white dark:bg-slate-800 rounded-xl shadow-md p-5 border border-slate-200 dark:border-slate-700">
          <h3 className="text-lg font-semibold text-slate-900 dark:text-white mb-3">模型滚动</h3>
          <div className="text-sm text-slate-700 dark:text-slate-300 space-y-1">
            <div>model: {String(rollout?.model_name || 'n/a')}:{String(rollout?.model_version || 'n/a')}</div>
            <div>stage: {num(rollout?.stage_pct, 0)}%</div>
            <div>status: {String(rollout?.status || 'unknown')}</div>
          </div>
        </div>

        <div className="bg-white dark:bg-slate-800 rounded-xl shadow-md p-5 border border-slate-200 dark:border-slate-700">
          <h3 className="text-lg font-semibold text-slate-900 dark:text-white mb-3">PnL (7d)</h3>
          <div className="text-sm text-slate-700 dark:text-slate-300 space-y-1">
            <div>net_pnl: {num(pnlTotals?.net_pnl, 0).toFixed(4)}</div>
            <div>fee: {num(pnlTotals?.fee, 0).toFixed(4)}</div>
            <div>slippage: {num(pnlTotals?.slippage, 0).toFixed(4)}</div>
            <div>impact: {num(pnlTotals?.impact, 0).toFixed(4)}</div>
          </div>
        </div>

        <div className="bg-white dark:bg-slate-800 rounded-xl shadow-md p-5 border border-slate-200 dark:border-slate-700">
          <h3 className="text-lg font-semibold text-slate-900 dark:text-white mb-3">持续模拟盘+训练</h3>
          <div className="text-sm text-slate-700 dark:text-slate-300 space-y-1">
            <div>cycle: {opsCycle}</div>
            <div>state: {String(opsState?.status || 'missing')}</div>
            <div>paper filled/rejected: {num(opsState?.paper?.filled, 0)}/{num(opsState?.paper?.rejected, 0)}</div>
            <div>train: {String(opsState?.train?.status || 'n/a')}</div>
            <div>backtest: {String(opsState?.backtest?.status || 'n/a')}</div>
          </div>
        </div>
      </div>

      <div className="bg-white dark:bg-slate-800 rounded-xl shadow-md p-6 border border-slate-200 dark:border-slate-700">
        <h3 className="text-lg font-semibold text-slate-900 dark:text-white mb-4">人工实盘控制台</h3>
        <div className="text-sm text-slate-700 dark:text-slate-300 mb-4 space-y-1">
          <div>live_enabled: {String(liveEnabled)}</div>
          <div>paper_enabled: {String(paperEnabled)}</div>
          <div>candidate: {String(latestCandidate?.model_name || 'n/a')}:{String(latestCandidate?.model_version || 'n/a')}</div>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-3">
          <button
            disabled={controlBusy}
            onClick={() => sendControl('/api/v2/control/live/disable', { paper_enabled: true, reason: 'manual_start_paper' }, 'paper trading started')}
            className="px-4 py-2 rounded-lg bg-blue-600 text-white hover:opacity-90 disabled:opacity-50"
          >
            启动模拟盘
          </button>
          <button
            disabled={controlBusy}
            onClick={() => sendControl('/api/v2/control/live/disable', { paper_enabled: false, reason: 'manual_stop_paper' }, 'paper trading stopped')}
            className="px-4 py-2 rounded-lg bg-slate-600 text-white hover:opacity-90 disabled:opacity-50"
          >
            关闭模拟盘
          </button>
          <button
            disabled={controlBusy}
            onClick={() => sendControl('/api/v2/control/model/switch-candidate', {
              track: 'liquid',
              model_name: String(latestCandidate?.model_name || rollout?.model_name || ''),
              model_version: String(latestCandidate?.model_version || rollout?.model_version || ''),
              reason: 'manual_switch_candidate',
            }, 'candidate switch requested')}
            className="px-4 py-2 rounded-lg bg-amber-600 text-white hover:opacity-90 disabled:opacity-50"
          >
            切换候选模型
          </button>
          <button
            disabled={controlBusy}
            onClick={() => sendControl('/api/v2/control/live/enable', { paper_enabled: false, reason: 'manual_enable_live' }, 'live trading enabled')}
            className="px-4 py-2 rounded-lg bg-red-600 text-white hover:opacity-90 disabled:opacity-50"
          >
            开启实盘
          </button>
          <button
            disabled={controlBusy}
            onClick={() => sendControl('/api/v2/control/live/disable', { paper_enabled: true, reason: 'manual_disable_live' }, 'live trading disabled, fallback to paper')}
            className="px-4 py-2 rounded-lg bg-green-700 text-white hover:opacity-90 disabled:opacity-50"
          >
            关闭实盘回退模拟盘
          </button>
        </div>
      </div>

      <div className="bg-white dark:bg-slate-800 rounded-xl shadow-md p-6 border border-slate-200 dark:border-slate-700">
        <h3 className="text-lg font-semibold text-slate-900 dark:text-white mb-4">服务详情</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {services.map(([name, st]) => (
            <div key={String(name)} className="bg-slate-50 dark:bg-slate-700/50 rounded-lg p-4 border border-slate-200 dark:border-slate-600">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-medium text-slate-700 dark:text-slate-300 capitalize">{String(name)}</span>
                <span className={`px-2 py-1 rounded text-xs font-semibold ${String(st) === 'running' ? 'bg-green-100 text-green-700 dark:bg-green-900 dark:text-green-300' : 'bg-red-100 text-red-700 dark:bg-red-900 dark:text-red-300'}`}>
                  {String(st)}
                </span>
              </div>
              <div className="flex items-center space-x-2 text-xs text-slate-600 dark:text-slate-400">
                {String(st) === 'running' ? <CheckCircle className="w-4 h-4 text-green-500" /> : <AlertTriangle className="w-4 h-4 text-red-500" />}
                <span>{String(st) === 'running' ? '正常运行' : '异常或停止'}</span>
              </div>
            </div>
          ))}
          {!services.length && (
            <div className="text-sm text-slate-500 dark:text-slate-400">
              {isLoading ? '加载中...' : '暂无服务状态'}
            </div>
          )}
        </div>
      </div>

      <div className="bg-white dark:bg-slate-800 rounded-xl shadow-md p-6 border border-slate-200 dark:border-slate-700">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-slate-900 dark:text-white">系统日志</h3>
          <span className="text-xs text-slate-500 dark:text-slate-400">最新 120 条</span>
        </div>
        <div className="bg-slate-950 dark:bg-slate-900 rounded-lg p-4 h-80 overflow-y-auto font-mono text-xs">
          {logs.map((log) => (
            <div key={log.id} className={`flex items-start space-x-2 mb-1.5 ${getLogColor(log.level)}`}>
              <span className="flex-shrink-0 mt-0.5">{getLogIcon(log.level)}</span>
              <span className="text-slate-400 dark:text-slate-500 select-none">{log.timestamp}</span>
              <span className="flex-1 break-all">{log.message}</span>
            </div>
          ))}
          {!logs.length && <div className="text-slate-500">No logs</div>}
        </div>
      </div>
    </div>
  );
}
