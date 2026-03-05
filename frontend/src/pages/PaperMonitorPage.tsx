import { useMemo, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import MetricLine from '../components/MetricLine'
import { apiGet, apiSend } from '../lib/api'

export default function PaperMonitorPage() {
  const qc = useQueryClient()
  const [symbol, setSymbol] = useState('BTC')
  const [paperUsdt, setPaperUsdt] = useState('500')

  const stateQ = useQuery({
    queryKey: ['live-state'],
    queryFn: () => apiGet<any>('/api/v2/control/live/state'),
    refetchInterval: 5000,
  })
  const perfQ = useQuery({
    queryKey: ['paper-performance'],
    queryFn: () => apiGet<any>('/api/v2/monitor/paper-performance'),
    refetchInterval: 10000,
  })
  const riskQ = useQuery({
    queryKey: ['paper-risk-events'],
    queryFn: () => apiGet<any>('/api/v2/risk/events/recent?limit=60'),
    refetchInterval: 5000,
  })
  const klineQ = useQuery({
    queryKey: ['paper-kline', symbol],
    queryFn: () => apiGet<any>(`/api/prices/${symbol}`),
    refetchInterval: 10000,
  })

  const savePaperCapital = useMutation({
    mutationFn: () =>
      apiSend<any>('/api/v2/config', 'PUT', {
        config_key: 'paper.initial_usdt',
        value_json: { value: Number(paperUsdt) },
        scope: 'global',
        requires_restart: false,
        description: 'paper initial balance',
      }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['live-state'] }),
  })

  const prices = Array.isArray(klineQ.data) ? klineQ.data : []
  const points = useMemo(
    () =>
      prices.slice(-200).map((x: any, i: number) => ({
        ts: String(x.timestamp || x.ts || i),
        value: Number(x.price || x.close || 0),
      })),
    [prices],
  )
  const events = Array.isArray(riskQ.data?.risk_events) ? riskQ.data.risk_events : []
  const paperEnabled = Boolean(stateQ.data?.paper_enabled)
  const perf = perfQ.data || {}

  return (
    <div className="space-y-4">
      <h1 className="text-2xl font-bold text-slate-800">模拟盘监管</h1>
      <div className="grid grid-cols-1 gap-4 md:grid-cols-4">
        <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
          <div className="text-xs text-slate-500">Paper Enabled</div>
          <div className={`text-2xl font-semibold ${paperEnabled ? 'text-emerald-700' : 'text-rose-700'}`}>{paperEnabled ? 'ON' : 'OFF'}</div>
        </div>
        <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
          <div className="text-xs text-slate-500">Equity</div>
          <div className="text-2xl font-semibold text-slate-900">{Number(perf?.equity || 0).toFixed(2)}</div>
        </div>
        <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
          <div className="text-xs text-slate-500">Free Margin</div>
          <div className="text-2xl font-semibold text-slate-900">{Number(perf?.free_margin || 0).toFixed(2)}</div>
        </div>
        <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
          <div className="text-xs text-slate-500">Reject Rate</div>
          <div className="text-2xl font-semibold text-slate-900">{(Number(perf?.reject_rate || 0) * 100).toFixed(2)}%</div>
        </div>
      </div>

      <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
        <div className="mb-2 text-sm font-semibold text-slate-700">热更新配置: 模拟盘初始资金 (默认 500 USDT)</div>
        <div className="flex items-center gap-2">
          <input value={paperUsdt} onChange={(e) => setPaperUsdt(e.target.value)} className="rounded border border-slate-300 px-3 py-2 text-sm" />
          <button onClick={() => savePaperCapital.mutate()} className="rounded bg-slate-900 px-4 py-2 text-sm text-white">
            保存并热生效
          </button>
        </div>
      </div>

      <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
        <div className="mb-2 flex items-center gap-2">
          <span className="text-sm font-semibold text-slate-700">K线与信号</span>
          <input value={symbol} onChange={(e) => setSymbol(e.target.value.toUpperCase())} className="w-24 rounded border border-slate-300 px-2 py-1 text-sm" />
        </div>
        <MetricLine title={`${symbol} price`} points={points} color="#0369a1" />
      </div>

      <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
        <div className="mb-2 text-sm font-semibold text-slate-700">近 30 天操作 / 风险事件</div>
        <div className="max-h-72 overflow-auto text-sm">
          {events.map((e: any) => (
            <div key={String(e.id)} className="border-b py-2">
              <div className="font-mono text-xs text-slate-500">{String(e.created_at || '')}</div>
              <div className="text-slate-700">
                {String(e.code)} / {String(e.message)}
              </div>
            </div>
          ))}
          {events.length === 0 && <div className="py-2 text-slate-400">no events</div>}
        </div>
      </div>
    </div>
  )
}
