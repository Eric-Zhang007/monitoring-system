import { useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { apiGet, apiSend } from '../lib/api'

export default function RiskCenterPage() {
  const qc = useQueryClient()
  const [cmd, setCmd] = useState('CMD KILL_SWITCH ON track=liquid strategy=global')
  const [source, setSource] = useState('frontend')

  const riskQ = useQuery({
    queryKey: ['risk-center-events'],
    queryFn: () => apiGet<any>('/api/v2/risk/events/recent?limit=100'),
    refetchInterval: 5000,
  })
  const cmdLogQ = useQuery({
    queryKey: ['risk-command-logs'],
    queryFn: () => apiGet<any>('/api/v2/risk/commands/logs?limit=100'),
    refetchInterval: 5000,
  })
  const killQ = useQuery({
    queryKey: ['risk-kill-status'],
    queryFn: () => apiGet<any>('/api/v2/risk/kill-switch?track=liquid&strategy_id=global'),
    refetchInterval: 5000,
  })
  const mailQ = useQuery({
    queryKey: ['mail-delivery-logs'],
    queryFn: () => apiGet<any>('/api/v2/ops/mail_delivery_logs?limit=100'),
    refetchInterval: 5000,
  })

  const runCmd = useMutation({
    mutationFn: () => apiSend<any>('/api/v2/risk/commands/execute', 'POST', { source, command: cmd }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['risk-center-events'] })
      qc.invalidateQueries({ queryKey: ['risk-command-logs'] })
      qc.invalidateQueries({ queryKey: ['risk-kill-status'] })
    },
  })

  const events = Array.isArray(riskQ.data?.risk_events) ? riskQ.data.risk_events : []
  const recon = Array.isArray(riskQ.data?.reconciliation_logs) ? riskQ.data.reconciliation_logs : []
  const logs = Array.isArray(cmdLogQ.data?.items) ? cmdLogQ.data.items : []
  const mails = Array.isArray(mailQ.data?.items) ? mailQ.data.items : []

  return (
    <div className="space-y-4">
      <h1 className="text-2xl font-bold text-slate-800">风控中心 + 邮件命令</h1>
      <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
        <div className="mb-1 text-sm text-slate-600">严格命令语法（成功/失败都会审计并邮件回执）</div>
        <div className="mb-2 text-xs text-slate-500">
          示例: CMD KILL_SWITCH ON track=liquid strategy=global / CMD PAUSE LIVE account=acc1 / CMD RESTART PROCESS id=...
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <select value={source} onChange={(e) => setSource(e.target.value)} className="rounded border px-3 py-2 text-sm">
            <option value="frontend">frontend</option>
            <option value="email">email</option>
          </select>
          <input value={cmd} onChange={(e) => setCmd(e.target.value)} className="min-w-[420px] rounded border px-3 py-2 text-sm" />
          <button onClick={() => runCmd.mutate()} className="rounded bg-slate-900 px-4 py-2 text-sm text-white">
            执行命令
          </button>
        </div>
        {runCmd.data && (
          <pre className="mt-2 overflow-auto rounded bg-slate-50 p-2 text-xs">{JSON.stringify(runCmd.data, null, 2)}</pre>
        )}
      </div>

      <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
        <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
          <div className="text-xs text-slate-500">Kill Switch</div>
          <div className={`text-2xl font-semibold ${killQ.data?.triggered ? 'text-rose-700' : 'text-emerald-700'}`}>{killQ.data?.triggered ? 'TRIGGERED' : 'ARMED'}</div>
          <div className="text-xs text-slate-500">{String(killQ.data?.reason || '')}</div>
        </div>
        <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
          <div className="text-xs text-slate-500">Risk Events</div>
          <div className="text-2xl font-semibold text-slate-900">{events.length}</div>
        </div>
        <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
          <div className="text-xs text-slate-500">Recon Logs</div>
          <div className="text-2xl font-semibold text-slate-900">{recon.length}</div>
        </div>
      </div>

      <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
        <div className="mb-2 text-sm font-semibold text-slate-700">邮件发送状态</div>
        <div className="max-h-52 overflow-auto text-xs">
          {mails.map((m: any) => (
            <div key={String(m.id)} className={`border-b py-2 ${m.send_ok ? 'text-slate-700' : 'text-rose-700'}`}>
              <div className="font-mono">{String(m.created_at || '')}</div>
              <div>{String(m.event_type)} / send_ok={String(m.send_ok)}</div>
              <div>{String(m.error || '')}</div>
            </div>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
          <div className="mb-2 text-sm font-semibold text-slate-700">risk_events</div>
          <div className="max-h-80 overflow-auto text-xs">
            {events.map((e: any) => (
              <div key={String(e.id)} className="border-b py-2">
                <div className="font-mono text-slate-500">{String(e.created_at || '')}</div>
                <div>{String(e.code)}: {String(e.message)}</div>
              </div>
            ))}
          </div>
        </div>
        <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
          <div className="mb-2 text-sm font-semibold text-slate-700">邮件/前端命令执行日志</div>
          <div className="max-h-80 overflow-auto text-xs">
            {logs.map((l: any) => (
              <div key={String(l.id)} className="border-b py-2">
                <div className="font-mono text-slate-500">{String(l.created_at || '')}</div>
                <div>{String(l.source)} / parse={String(l.parse_ok)} / execute={String(l.execute_ok)}</div>
                <div>{String(l.command_text)}</div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}
