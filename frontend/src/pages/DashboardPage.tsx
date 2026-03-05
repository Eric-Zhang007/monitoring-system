import { useQuery } from '@tanstack/react-query'
import { apiGet } from '../lib/api'

export default function DashboardPage() {
  const statusQ = useQuery({
    queryKey: ['api-status'],
    queryFn: () => apiGet<any>('/api/status'),
    refetchInterval: 15000,
  })
  const procQ = useQuery({
    queryKey: ['ops-processes'],
    queryFn: () => apiGet<any>('/api/v2/ops/processes'),
    refetchInterval: 5000,
  })
  const riskQ = useQuery({
    queryKey: ['risk-events'],
    queryFn: () => apiGet<any>('/api/v2/risk/events/recent?limit=20'),
    refetchInterval: 5000,
  })
  const connQ = useQuery({
    queryKey: ['connectivity'],
    queryFn: () => apiGet<any>('/api/v2/ops/connectivity?limit=20'),
    refetchInterval: 10000,
  })
  const clockQ = useQuery({
    queryKey: ['clock-drift'],
    queryFn: () => apiGet<any>('/api/v2/ops/clock_drift'),
    refetchInterval: 10000,
  })

  const processes = Array.isArray(procQ.data?.items) ? procQ.data.items : []
  const running = processes.filter((x: any) => String(x?.status) === 'running').length
  const riskEvents = Array.isArray(riskQ.data?.risk_events) ? riskQ.data.risk_events : []
  const conns = Array.isArray(connQ.data?.items) ? connQ.data.items : []
  const latestBitget = conns.find((x: any) => String(x?.venue) === 'bitget') || conns[0] || {}
  const clock = clockQ.data || {}
  const gpus = statusQ.data?.system?.gpu?.gpus || []

  return (
    <div className="space-y-4">
      <h1 className="text-2xl font-bold text-slate-800">控制台总览</h1>
      <div className="grid grid-cols-1 gap-4 md:grid-cols-4">
        <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
          <div className="text-xs text-slate-500">运行中进程</div>
          <div className="text-2xl font-semibold text-slate-900">{running}</div>
        </div>
        <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
          <div className="text-xs text-slate-500">最近风险事件</div>
          <div className="text-2xl font-semibold text-rose-700">{riskEvents.length}</div>
        </div>
        <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
          <div className="text-xs text-slate-500">服务器成本</div>
          <div className="text-2xl font-semibold text-emerald-700">1.88 CNY / hour</div>
        </div>
        <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
          <div className="text-xs text-slate-500">GPU 数量</div>
          <div className="text-2xl font-semibold text-slate-900">{Array.isArray(gpus) ? gpus.length : 0}</div>
        </div>
      </div>

      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
          <div className="text-xs text-slate-500">交易所网络可达性</div>
          <div className={`text-2xl font-semibold ${latestBitget?.rest_ok && latestBitget?.ws_ok ? 'text-emerald-700' : 'text-rose-700'}`}>
            {latestBitget?.rest_ok && latestBitget?.ws_ok ? 'REST+WS OK' : 'UNREACHABLE'}
          </div>
          <div className="text-xs text-slate-500">
            venue={String(latestBitget?.venue || '-')} proxy={String(latestBitget?.using_proxy_profile || '-')}
          </div>
        </div>
        <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
          <div className="text-xs text-slate-500">UTC 时钟漂移</div>
          <div className={`text-2xl font-semibold ${String(clock?.level || '').toLowerCase() === 'red' ? 'text-rose-700' : String(clock?.level || '').toLowerCase() === 'yellow' ? 'text-amber-700' : 'text-emerald-700'}`}>
            {String(clock?.level || 'unknown').toUpperCase()}
          </div>
          <div className="text-xs text-slate-500">
            drift_ms={clock?.drift_ms ?? '-'} local={String(clock?.local_utc || '-')}
          </div>
        </div>
      </div>

      <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
        <h2 className="mb-3 text-lg font-semibold text-slate-800">进程状态</h2>
        <div className="overflow-x-auto">
          <table className="min-w-full text-sm">
            <thead>
              <tr className="border-b text-left text-slate-500">
                <th className="py-2">ID</th>
                <th>Type</th>
                <th>Status</th>
                <th>PID</th>
                <th>Track</th>
              </tr>
            </thead>
            <tbody>
              {processes.map((p: any) => (
                <tr key={String(p.process_id)} className="border-b">
                  <td className="py-2 font-mono text-xs">{String(p.process_id)}</td>
                  <td>{String(p.task_type)}</td>
                  <td>{String(p.status)}</td>
                  <td>{String(p.pid ?? '-')}</td>
                  <td>{String(p.track ?? '-')}</td>
                </tr>
              ))}
              {processes.length === 0 && (
                <tr>
                  <td colSpan={5} className="py-4 text-center text-slate-400">
                    no process
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}
