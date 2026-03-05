import { useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { apiGet, apiSend } from '../lib/api'

export default function ConfigCenterPage() {
  const qc = useQueryClient()
  const [form, setForm] = useState({
    config_key: '',
    value: '',
    scope: 'global',
    scope_id: '',
    requires_restart: false,
    description: '',
  })
  const [restartProcessId, setRestartProcessId] = useState('')
  const [proxyForm, setProxyForm] = useState({
    profile_id: '',
    name: '',
    proxy_type: 'http',
    host: '127.0.0.1',
    port: '7890',
    username: '',
    password: '',
    enabled: true,
    note: '',
  })
  const [probeProxyId, setProbeProxyId] = useState('')

  const cfgQ = useQuery({
    queryKey: ['runtime-config'],
    queryFn: () => apiGet<any>('/api/v2/config'),
    refetchInterval: 10000,
  })
  const auditQ = useQuery({
    queryKey: ['runtime-config-audit'],
    queryFn: () => apiGet<any>('/api/v2/config/audit?limit=50'),
    refetchInterval: 10000,
  })
  const procQ = useQuery({
    queryKey: ['process-list'],
    queryFn: () => apiGet<any>('/api/v2/ops/processes'),
    refetchInterval: 5000,
  })
  const proxyQ = useQuery({
    queryKey: ['proxy-profiles'],
    queryFn: () => apiGet<any>('/api/v2/ops/proxy_profiles'),
    refetchInterval: 10000,
  })
  const connQ = useQuery({
    queryKey: ['connectivity-status'],
    queryFn: () => apiGet<any>('/api/v2/ops/connectivity?limit=20'),
    refetchInterval: 10000,
  })
  const saveCfg = useMutation({
    mutationFn: () =>
      apiSend<any>('/api/v2/config', 'PUT', {
        config_key: form.config_key,
        value_json: { value: form.value },
        scope: form.scope,
        scope_id: form.scope_id,
        requires_restart: form.requires_restart,
        description: form.description,
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['runtime-config'] })
      qc.invalidateQueries({ queryKey: ['runtime-config-audit'] })
    },
  })
  const restartProc = useMutation({
    mutationFn: () => apiSend<any>('/api/v2/ops/process/restart', 'POST', { process_id: restartProcessId }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['process-list'] }),
  })
  const saveProxy = useMutation({
    mutationFn: () =>
      apiSend<any>('/api/v2/ops/proxy_profiles', 'POST', {
        profile_id: proxyForm.profile_id || undefined,
        name: proxyForm.name || proxyForm.profile_id,
        proxy_type: proxyForm.proxy_type,
        host: proxyForm.host,
        port: Number(proxyForm.port),
        username: proxyForm.username || undefined,
        password: proxyForm.password || undefined,
        enabled: proxyForm.enabled,
        note: proxyForm.note,
      }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['proxy-profiles'] }),
  })
  const probeConn = useMutation({
    mutationFn: () => apiSend<any>('/api/v2/ops/connectivity/probe', 'POST', { venue: 'bitget', proxy_profile_id: probeProxyId || undefined }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['connectivity-status'] }),
  })

  const cfgRows = Array.isArray(cfgQ.data?.items) ? cfgQ.data.items : []
  const audits = Array.isArray(auditQ.data?.items) ? auditQ.data.items : []
  const procs = Array.isArray(procQ.data?.items) ? procQ.data.items : []
  const proxies = Array.isArray(proxyQ.data?.items) ? proxyQ.data.items : []
  const connRows = Array.isArray(connQ.data?.items) ? connQ.data.items : []

  return (
    <div className="space-y-4">
      <h1 className="text-2xl font-bold text-slate-800">配置中心</h1>
      <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
        <div className="mb-2 text-sm font-semibold text-slate-700">更新配置 (热更新)</div>
        <div className="grid grid-cols-1 gap-2 md:grid-cols-3">
          <input value={form.config_key} onChange={(e) => setForm((s) => ({ ...s, config_key: e.target.value }))} className="rounded border px-3 py-2 text-sm" placeholder="config_key" />
          <input value={form.value} onChange={(e) => setForm((s) => ({ ...s, value: e.target.value }))} className="rounded border px-3 py-2 text-sm" placeholder="value" />
          <input value={form.scope_id} onChange={(e) => setForm((s) => ({ ...s, scope_id: e.target.value }))} className="rounded border px-3 py-2 text-sm" placeholder="scope_id(optional)" />
          <select value={form.scope} onChange={(e) => setForm((s) => ({ ...s, scope: e.target.value }))} className="rounded border px-3 py-2 text-sm">
            <option value="global">global</option>
            <option value="track">track</option>
            <option value="process">process</option>
            <option value="account">account</option>
          </select>
          <input value={form.description} onChange={(e) => setForm((s) => ({ ...s, description: e.target.value }))} className="rounded border px-3 py-2 text-sm" placeholder="description" />
          <label className="flex items-center gap-2 text-sm">
            <input type="checkbox" checked={form.requires_restart} onChange={(e) => setForm((s) => ({ ...s, requires_restart: e.target.checked }))} />
            requires_restart (需要重启)
          </label>
        </div>
        <button onClick={() => saveCfg.mutate()} className="mt-3 rounded bg-slate-900 px-4 py-2 text-sm text-white">
          保存配置
        </button>
      </div>

      <div className="rounded-xl border border-amber-200 bg-amber-50 p-4 text-sm text-amber-900">
        对 `requires_restart=true` 的配置，前端会标注“需要重启”。可在下方选择进程并执行一键重启。
      </div>

      <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
        <div className="mb-2 text-sm font-semibold text-slate-700">一键重启进程</div>
        <div className="flex flex-wrap items-center gap-2">
          <select value={restartProcessId} onChange={(e) => setRestartProcessId(e.target.value)} className="rounded border px-3 py-2 text-sm">
            <option value="">select process</option>
            {procs.map((p: any) => (
              <option key={String(p.process_id)} value={String(p.process_id)}>
                {String(p.process_id)} / {String(p.task_type)} / {String(p.status)}
              </option>
            ))}
          </select>
          <button onClick={() => restartProc.mutate()} className="rounded bg-emerald-700 px-4 py-2 text-sm text-white">
            重启
          </button>
        </div>
      </div>

      <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
        <div className="mb-2 text-sm font-semibold text-slate-700">当前配置</div>
        <div className="max-h-64 overflow-auto">
          <table className="min-w-full text-sm">
            <thead>
              <tr className="border-b text-left text-slate-500">
                <th className="py-2">Key</th>
                <th>Scope</th>
                <th>Value</th>
                <th>Version</th>
                <th>Restart</th>
              </tr>
            </thead>
            <tbody>
              {cfgRows.map((c: any) => (
                <tr key={`${String(c.scope)}:${String(c.scope_id)}:${String(c.config_key)}`} className="border-b">
                  <td className="py-2 font-mono text-xs">{String(c.config_key)}</td>
                  <td>{String(c.scope)} / {String(c.scope_id || '-')}</td>
                  <td className="font-mono text-xs">{JSON.stringify(c.value_json)}</td>
                  <td>{String(c.version)}</td>
                  <td>{c.requires_restart ? 'YES' : 'NO'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
        <div className="mb-2 text-sm font-semibold text-slate-700">配置审计日志</div>
        <div className="max-h-64 overflow-auto text-xs">
          {audits.map((a: any) => (
            <div key={String(a.id)} className="border-b py-2">
              <div className="font-mono text-slate-500">{String(a.created_at)}</div>
              <div>{String(a.config_key)} / {String(a.scope)} / {String(a.scope_id || '-')}</div>
              <div>old={JSON.stringify(a.old_value_json)} new={JSON.stringify(a.new_value_json)}</div>
              <div>by={String(a.updated_by)}</div>
            </div>
          ))}
        </div>
      </div>

      <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
        <div className="mb-2 text-sm font-semibold text-slate-700">Proxy Profile 管理</div>
        <div className="grid grid-cols-1 gap-2 md:grid-cols-3">
          <input value={proxyForm.profile_id} onChange={(e) => setProxyForm((s) => ({ ...s, profile_id: e.target.value }))} className="rounded border px-3 py-2 text-sm" placeholder="profile_id" />
          <input value={proxyForm.name} onChange={(e) => setProxyForm((s) => ({ ...s, name: e.target.value }))} className="rounded border px-3 py-2 text-sm" placeholder="name" />
          <select value={proxyForm.proxy_type} onChange={(e) => setProxyForm((s) => ({ ...s, proxy_type: e.target.value }))} className="rounded border px-3 py-2 text-sm">
            <option value="http">http</option>
            <option value="socks5">socks5</option>
          </select>
          <input value={proxyForm.host} onChange={(e) => setProxyForm((s) => ({ ...s, host: e.target.value }))} className="rounded border px-3 py-2 text-sm" placeholder="host" />
          <input value={proxyForm.port} onChange={(e) => setProxyForm((s) => ({ ...s, port: e.target.value }))} className="rounded border px-3 py-2 text-sm" placeholder="port" />
          <input value={proxyForm.username} onChange={(e) => setProxyForm((s) => ({ ...s, username: e.target.value }))} className="rounded border px-3 py-2 text-sm" placeholder="username(optional)" />
          <input value={proxyForm.password} onChange={(e) => setProxyForm((s) => ({ ...s, password: e.target.value }))} className="rounded border px-3 py-2 text-sm" placeholder="password(write-only)" />
          <input value={proxyForm.note} onChange={(e) => setProxyForm((s) => ({ ...s, note: e.target.value }))} className="rounded border px-3 py-2 text-sm" placeholder="note" />
          <label className="flex items-center gap-2 text-sm">
            <input type="checkbox" checked={proxyForm.enabled} onChange={(e) => setProxyForm((s) => ({ ...s, enabled: e.target.checked }))} />
            enabled
          </label>
        </div>
        <div className="mt-2 flex flex-wrap items-center gap-2">
          <button onClick={() => saveProxy.mutate()} className="rounded bg-slate-900 px-4 py-2 text-sm text-white">
            保存 Proxy Profile
          </button>
          <select value={probeProxyId} onChange={(e) => setProbeProxyId(e.target.value)} className="rounded border px-3 py-2 text-sm">
            <option value="">probe without proxy</option>
            {proxies.map((p: any) => (
              <option key={String(p.profile_id)} value={String(p.profile_id)}>
                {String(p.profile_id)} / {String(p.proxy_type)} / {String(p.host)}:{String(p.port)}
              </option>
            ))}
          </select>
          <button onClick={() => probeConn.mutate()} className="rounded bg-emerald-700 px-4 py-2 text-sm text-white">
            执行 Bitget 连通性探测
          </button>
        </div>
        <div className="mt-3 max-h-56 overflow-auto text-xs">
          {proxies.map((p: any) => (
            <div key={String(p.profile_id)} className="border-b py-2">
              <div>{String(p.profile_id)} / {String(p.name)} / {String(p.proxy_type)}://{String(p.host)}:{String(p.port)}</div>
              <div>enabled={String(p.enabled)} note={String(p.note || '')}</div>
            </div>
          ))}
        </div>
      </div>

      <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
        <div className="mb-2 text-sm font-semibold text-slate-700">Connectivity 历史</div>
        <div className="max-h-56 overflow-auto text-xs">
          {connRows.map((c: any, idx: number) => (
            <div key={`${String(c.id || idx)}:${String(c.ts || idx)}`} className="border-b py-2">
              <div>{String(c.ts || '')} / venue={String(c.venue)} / rest={String(c.rest_ok)} ws={String(c.ws_ok)}</div>
              <div>proxy={String(c.using_proxy_profile || '-')} err={String(c.error || '')}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
