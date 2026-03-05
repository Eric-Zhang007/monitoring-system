import { useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { apiGet, apiSend } from '../lib/api'

export default function LiveMonitorPage() {
  const qc = useQueryClient()
  const [form, setForm] = useState({
    account_id: '',
    account_name: '',
    api_key: '',
    api_secret: '',
    passphrase: '',
  })
  const [connectivityAccount, setConnectivityAccount] = useState<string>('default')

  const accQ = useQuery({
    queryKey: ['live-accounts'],
    queryFn: () => apiGet<any>('/api/v2/live/accounts'),
    refetchInterval: 10000,
  })
  const killQ = useQuery({
    queryKey: ['live-kill-switch'],
    queryFn: () => apiGet<any>('/api/v2/risk/kill-switch?track=liquid&strategy_id=global'),
    refetchInterval: 5000,
  })
  const connQ = useQuery({
    queryKey: ['live-connectivity', connectivityAccount],
    queryFn: () => apiGet<any>(`/api/v2/live/accounts/${connectivityAccount}/connectivity`),
    refetchInterval: 10000,
  })

  const saveAccount = useMutation({
    mutationFn: () => apiSend<any>('/api/v2/live/accounts', 'POST', form),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['live-accounts'] })
      setForm({ account_id: '', account_name: '', api_key: '', api_secret: '', passphrase: '' })
    },
  })

  const startLive = useMutation({
    mutationFn: (accountId: string) =>
      apiSend<any>('/api/v2/ops/process/start', 'POST', {
        task_type: 'LIVE_TRADER',
        params: { account_id: accountId, track: 'liquid', task_type: 'LIVE_TRADER' },
      }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['live-accounts'] }),
  })

  const accounts = Array.isArray(accQ.data?.items) ? accQ.data.items : []
  const kill = killQ.data || {}

  return (
    <div className="space-y-4">
      <h1 className="text-2xl font-bold text-slate-800">实盘监管 (Bitget 多账号)</h1>
      <div className="rounded-xl border border-amber-200 bg-amber-50 p-4 text-sm text-amber-900">
        此系统不提供提币/转账能力，不会调用 withdraw/transfer API。
      </div>

      <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
        <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
          <div className="text-xs text-slate-500">Kill Switch</div>
          <div className={`text-2xl font-semibold ${kill?.triggered ? 'text-rose-700' : 'text-emerald-700'}`}>{kill?.triggered ? 'TRIGGERED' : 'ARMED'}</div>
          <div className="text-xs text-slate-500">{String(kill?.reason || '')}</div>
        </div>
        <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
          <div className="text-xs text-slate-500">Connectivity Probe</div>
          <div className={`text-2xl font-semibold ${connQ.data?.reachable ? 'text-emerald-700' : 'text-rose-700'}`}>{connQ.data?.reachable ? 'REACHABLE' : 'UNREACHABLE'}</div>
          <div className="text-xs text-slate-500">{String(connQ.data?.detail || '')}</div>
        </div>
        <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
          <div className="text-xs text-slate-500">Env Default Account</div>
          <div className="text-2xl font-semibold text-slate-900">{accQ.data?.default_env_account?.configured ? 'CONFIGURED' : 'MISSING'}</div>
        </div>
      </div>

      <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
        <div className="mb-2 text-sm font-semibold text-slate-700">添加/更新账号</div>
        <div className="grid grid-cols-1 gap-2 md:grid-cols-2">
          <input value={form.account_id} onChange={(e) => setForm((s) => ({ ...s, account_id: e.target.value }))} placeholder="account_id" className="rounded border px-3 py-2 text-sm" />
          <input value={form.account_name} onChange={(e) => setForm((s) => ({ ...s, account_name: e.target.value }))} placeholder="account_name" className="rounded border px-3 py-2 text-sm" />
          <input value={form.api_key} onChange={(e) => setForm((s) => ({ ...s, api_key: e.target.value }))} placeholder="api_key" className="rounded border px-3 py-2 text-sm" />
          <input value={form.api_secret} onChange={(e) => setForm((s) => ({ ...s, api_secret: e.target.value }))} placeholder="api_secret" className="rounded border px-3 py-2 text-sm" />
          <input value={form.passphrase} onChange={(e) => setForm((s) => ({ ...s, passphrase: e.target.value }))} placeholder="passphrase" className="rounded border px-3 py-2 text-sm" />
        </div>
        <button onClick={() => saveAccount.mutate()} className="mt-3 rounded bg-slate-900 px-4 py-2 text-sm text-white">
          保存账号密钥
        </button>
      </div>

      <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
        <div className="mb-2 text-sm font-semibold text-slate-700">账号列表</div>
        <table className="min-w-full text-sm">
          <thead>
            <tr className="border-b text-left text-slate-500">
              <th className="py-2">Account</th>
              <th>Name</th>
              <th>Enabled</th>
              <th>Default</th>
              <th>Ops</th>
            </tr>
          </thead>
          <tbody>
            {accounts.map((a: any) => (
              <tr key={String(a.account_id)} className="border-b">
                <td className="py-2 font-mono text-xs">{String(a.account_id)}</td>
                <td>{String(a.account_name)}</td>
                <td>{String(a.enabled)}</td>
                <td>{String(a.is_default)}</td>
                <td>
                  <button
                    onClick={() => setConnectivityAccount(String(a.account_id))}
                    className="mr-2 rounded bg-slate-100 px-2 py-1 text-xs"
                  >
                    Probe
                  </button>
                  <button onClick={() => startLive.mutate(String(a.account_id))} className="rounded bg-emerald-700 px-2 py-1 text-xs text-white">
                    Start Live
                  </button>
                </td>
              </tr>
            ))}
            {accounts.length === 0 && (
              <tr>
                <td colSpan={5} className="py-3 text-center text-slate-400">
                  no accounts
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  )
}
