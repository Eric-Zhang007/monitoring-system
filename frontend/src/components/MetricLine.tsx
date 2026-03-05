import { Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'

type Point = { ts: string; value: number }

export default function MetricLine({ title, points, color = '#0f766e' }: { title: string; points: Point[]; color?: string }) {
  return (
    <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
      <div className="mb-2 text-sm font-semibold text-slate-700">{title}</div>
      <div className="h-40">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={points}>
            <XAxis dataKey="ts" hide />
            <YAxis width={34} />
            <Tooltip />
            <Line type="monotone" dataKey="value" stroke={color} dot={false} strokeWidth={2} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}
