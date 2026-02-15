import { useState } from 'react'

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000'

export default function V2Lab() {
  const [log, setLog] = useState<string>('')
  const [company, setCompany] = useState('Test Startup')
  const [symbol, setSymbol] = useState('BTC')

  const append = (text: string) => setLog((prev) => `${prev}\n${text}`.trim())

  const callApi = async (path: string, init?: RequestInit) => {
    const res = await fetch(`${API_BASE}${path}`, {
      headers: { 'Content-Type': 'application/json' },
      ...init,
    })
    const body = await res.text()
    append(`${init?.method || 'GET'} ${path} -> ${res.status}`)
    append(body)
    append('---')
  }

  const seedEvent = async () => {
    const payload = {
      events: [
        {
          event_type: 'funding',
          title: `${company} raised Series A`,
          occurred_at: new Date().toISOString(),
          source_url: 'https://example.com/funding',
          source_name: 'manual-seed',
          source_timezone: 'UTC',
          source_tier: 2,
          confidence_score: 0.9,
          payload: { amount_usd: 12000000 },
          entities: [
            { entity_type: 'company', name: company, country: 'US', sector: 'AI', metadata: {} },
            { entity_type: 'investor', name: 'Test Capital', country: 'US', sector: 'VC', metadata: {} },
          ],
        },
      ],
    }

    await callApi('/api/v2/ingest/events', {
      method: 'POST',
      body: JSON.stringify(payload),
    })
  }

  const predictVC = async () => {
    await callApi('/api/v2/predict/vc', {
      method: 'POST',
      body: JSON.stringify({ company_name: company, horizon_months: 12 }),
    })
  }

  const predictLiquid = async () => {
    await callApi('/api/v2/predict/liquid', {
      method: 'POST',
      body: JSON.stringify({ symbol, horizon: '1d' }),
    })
  }

  return (
    <div className="space-y-4">
      <div className="bg-white dark:bg-slate-800 rounded-lg shadow p-4 border border-slate-200 dark:border-slate-700">
        <h3 className="text-lg font-semibold text-slate-900 dark:text-white mb-3">V2 API Lab</h3>
        <div className="grid gap-3 md:grid-cols-2">
          <input
            value={company}
            onChange={(e) => setCompany(e.target.value)}
            className="px-3 py-2 rounded border border-slate-300 dark:border-slate-600 bg-slate-50 dark:bg-slate-700"
            placeholder="Company"
          />
          <input
            value={symbol}
            onChange={(e) => setSymbol(e.target.value.toUpperCase())}
            className="px-3 py-2 rounded border border-slate-300 dark:border-slate-600 bg-slate-50 dark:bg-slate-700"
            placeholder="Symbol"
          />
        </div>
        <div className="mt-3 flex flex-wrap gap-2">
          <button onClick={seedEvent} className="px-3 py-2 rounded bg-blue-600 text-white">Seed Event</button>
          <button onClick={predictVC} className="px-3 py-2 rounded bg-emerald-600 text-white">Predict VC</button>
          <button onClick={predictLiquid} className="px-3 py-2 rounded bg-orange-600 text-white">Predict Liquid</button>
        </div>
      </div>

      <pre className="bg-slate-900 text-slate-100 rounded-lg p-4 text-xs overflow-auto max-h-[420px]">{log || 'No requests yet.'}</pre>
    </div>
  )
}
