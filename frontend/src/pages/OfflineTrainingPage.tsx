import { useMemo, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import MetricLine from '../components/MetricLine'
import { useSSE } from '../hooks/useSSE'
import { API_BASE, apiGet, apiSend } from '../lib/api'

export default function OfflineTrainingPage() {
  const qc = useQueryClient()
  const [auditTaskId, setAuditTaskId] = useState<string | null>(null)
  const [symbols, setSymbols] = useState('BTC,ETH,SOL')
  const runQ = useQuery({
    queryKey: ['offline-audit-latest'],
    queryFn: () => apiGet<any>('/api/v2/audit/offline_data?track=liquid'),
    refetchInterval: 10000,
  })
  const procQ = useQuery({
    queryKey: ['training-processes'],
    queryFn: () => apiGet<any>('/api/v2/ops/processes?task_type=TRAIN_LIQUID'),
    refetchInterval: 5000,
  })
  const perfQ = useQuery({
    queryKey: ['horizon-performance'],
    queryFn: () => apiGet<any>('/api/v2/monitor/horizon-performance'),
    refetchInterval: 15000,
  })
  const runAudit = useMutation({
    mutationFn: () => apiSend<any>('/api/v2/audit/offline_data/run', 'POST', { track: 'liquid', symbols }),
    onSuccess: (res) => {
      setAuditTaskId(String(res.task_id))
      qc.invalidateQueries({ queryKey: ['offline-audit-latest'] })
    },
  })
  const startTrain = useMutation({
    mutationFn: () =>
      apiSend<any>('/api/v2/ops/process/start', 'POST', {
        task_type: 'TRAIN_LIQUID',
        params: { symbols, task_type: 'TRAIN_LIQUID' },
      }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['training-processes'] }),
  })

  const sse = useSSE<any>(auditTaskId ? `${API_BASE}/api/v2/audit/offline_data/stream/${auditTaskId}` : null)
  const latest = runQ.data?.result || {}
  const gates = latest?.gates || runQ.data?.result?.gates || { ready: false, reasons: [] }
  const reasons = Array.isArray(gates.reasons) ? gates.reasons : []
  const trainRows = Array.isArray(procQ.data?.items) ? procQ.data.items : []
  const line = useMemo(() => {
    const items = perfQ.data?.horizons || {}
    return Object.keys(items).map((k) => ({ ts: k, value: Number(items?.[k]?.hit_rate || 0) }))
  }, [perfQ.data])

  return (
    <div className="space-y-4">
      <h1 className="text-2xl font-bold text-slate-800">离线训练监管</h1>
      <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
        <div className="mb-3 text-sm text-slate-600">先执行离线数据审计，再启动训练任务。strict-only 下 gate 不通过不建议训练。</div>
        <div className="flex flex-wrap items-center gap-2">
          <input
            value={symbols}
            onChange={(e) => setSymbols(e.target.value)}
            className="rounded border border-slate-300 px-3 py-2 text-sm"
            placeholder="BTC,ETH,SOL"
          />
          <button onClick={() => runAudit.mutate()} className="rounded bg-slate-900 px-4 py-2 text-sm text-white">
            运行离线审计
          </button>
          <button onClick={() => startTrain.mutate()} className="rounded bg-emerald-700 px-4 py-2 text-sm text-white">
            启动训练
          </button>
        </div>
        {auditTaskId && (
          <div className="mt-2 text-xs text-slate-500">
            task={auditTaskId} stream={sse.connected ? 'connected' : sse.error || 'idle'}
          </div>
        )}
      </div>

      <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
        <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
          <div className="text-xs text-slate-500">Audit Gate</div>
          <div className={`text-2xl font-semibold ${gates.ready ? 'text-emerald-700' : 'text-rose-700'}`}>{gates.ready ? 'READY' : 'BLOCKED'}</div>
        </div>
        <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
          <div className="text-xs text-slate-500">运行中训练任务</div>
          <div className="text-2xl font-semibold text-slate-900">{trainRows.filter((x: any) => x.status === 'running').length}</div>
        </div>
        <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
          <div className="text-xs text-slate-500">Artifacts</div>
          <div className="text-sm font-medium text-slate-800">artifacts/models/liquid_main</div>
        </div>
      </div>

      {reasons.length > 0 && (
        <div className="rounded-xl border border-rose-200 bg-rose-50 p-4 text-sm text-rose-800">
          <div className="mb-1 font-semibold">Gate Reasons</div>
          <ul className="list-disc pl-5">
            {reasons.map((r: string) => (
              <li key={r}>{r}</li>
            ))}
          </ul>
        </div>
      )}

      <MetricLine title="盈利/命中代理曲线 (horizon-performance)" points={line} color="#b45309" />
    </div>
  )
}
