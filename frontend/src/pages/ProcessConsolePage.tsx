import { useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { apiGet, apiSend } from '../lib/api'

export default function ProcessConsolePage() {
  const qc = useQueryClient()
  const [taskType, setTaskType] = useState('TRAIN_LIQUID')
  const [symbols, setSymbols] = useState('BTC,ETH,SOL')
  const [activePid, setActivePid] = useState('')

  const procQ = useQuery({
    queryKey: ['ops-processes-all'],
    queryFn: () => apiGet<any>('/api/v2/ops/processes'),
    refetchInterval: 3000,
  })
  const logQ = useQuery({
    queryKey: ['ops-process-logs', activePid],
    queryFn: () => (activePid ? apiGet<any>(`/api/v2/ops/process/${activePid}/logs?lines=200`) : Promise.resolve({ lines: [] })),
    refetchInterval: activePid ? 2000 : false,
  })
  const metricsQ = useQuery({
    queryKey: ['ops-process-metrics', activePid],
    queryFn: () => (activePid ? apiGet<any>(`/api/v2/ops/process/${activePid}/metrics`) : Promise.resolve({})),
    refetchInterval: activePid ? 2000 : false,
  })

  const startM = useMutation({
    mutationFn: () =>
      apiSend<any>('/api/v2/ops/process/start', 'POST', {
        task_type: taskType,
        params: { symbols, task_type: taskType },
      }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['ops-processes-all'] }),
  })
  const stopM = useMutation({
    mutationFn: (pid: string) => apiSend<any>('/api/v2/ops/process/stop', 'POST', { process_id: pid }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['ops-processes-all'] }),
  })
  const restartM = useMutation({
    mutationFn: (pid: string) => apiSend<any>('/api/v2/ops/process/restart', 'POST', { process_id: pid }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['ops-processes-all'] }),
  })

  const processes = Array.isArray(procQ.data?.items) ? procQ.data.items : []
  const lines = Array.isArray(logQ.data?.lines) ? logQ.data.lines : []

  return (
    <div className="space-y-4">
      <h1 className="text-2xl font-bold text-slate-800">进程控制台</h1>
      <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
        <div className="mb-2 text-sm font-semibold text-slate-700">启动新进程</div>
        <div className="flex flex-wrap items-center gap-2">
          <select value={taskType} onChange={(e) => setTaskType(e.target.value)} className="rounded border px-3 py-2 text-sm">
            <option>TRAIN_LIQUID</option>
            <option>TRAIN_VC</option>
            <option>PAPER_TRADER</option>
            <option>LIVE_TRADER</option>
            <option>RECON_DAEMON</option>
            <option>CONTINUOUS_FINETUNE</option>
            <option>ABLATION_EXPERIMENT</option>
          </select>
          <input value={symbols} onChange={(e) => setSymbols(e.target.value)} className="rounded border px-3 py-2 text-sm" placeholder="symbols" />
          <button onClick={() => startM.mutate()} className="rounded bg-slate-900 px-4 py-2 text-sm text-white">
            Start
          </button>
        </div>
      </div>

      <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
        <div className="mb-2 text-sm font-semibold text-slate-700">进程列表</div>
        <table className="min-w-full text-sm">
          <thead>
            <tr className="border-b text-left text-slate-500">
              <th className="py-2">Process ID</th>
              <th>Task</th>
              <th>Status</th>
              <th>PID</th>
              <th>Ops</th>
            </tr>
          </thead>
          <tbody>
            {processes.map((p: any) => {
              const pid = String(p.process_id)
              return (
                <tr key={pid} className="border-b">
                  <td className="py-2 font-mono text-xs">{pid}</td>
                  <td>{String(p.task_type)}</td>
                  <td>{String(p.status)}</td>
                  <td>{String(p.pid ?? '-')}</td>
                  <td className="space-x-2">
                    <button onClick={() => setActivePid(pid)} className="rounded bg-slate-100 px-2 py-1 text-xs">
                      View
                    </button>
                    <button onClick={() => restartM.mutate(pid)} className="rounded bg-amber-600 px-2 py-1 text-xs text-white">
                      Restart
                    </button>
                    <button onClick={() => stopM.mutate(pid)} className="rounded bg-rose-700 px-2 py-1 text-xs text-white">
                      Stop
                    </button>
                  </td>
                </tr>
              )
            })}
            {processes.length === 0 && (
              <tr>
                <td colSpan={5} className="py-3 text-center text-slate-400">
                  no process
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      {activePid && (
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
          <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
            <div className="mb-2 text-sm font-semibold text-slate-700">实时指标: {activePid}</div>
            <pre className="overflow-auto text-xs">{JSON.stringify(metricsQ.data || {}, null, 2)}</pre>
          </div>
          <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
            <div className="mb-2 text-sm font-semibold text-slate-700">日志 Tail: {activePid}</div>
            <pre className="max-h-80 overflow-auto text-xs">{lines.join('\n')}</pre>
          </div>
        </div>
      )}
    </div>
  )
}
