import { NavLink, Navigate, Route, Routes } from 'react-router-dom'
import ConfigCenterPage from './pages/ConfigCenterPage'
import DashboardPage from './pages/DashboardPage'
import LiveMonitorPage from './pages/LiveMonitorPage'
import OfflineTrainingPage from './pages/OfflineTrainingPage'
import PaperMonitorPage from './pages/PaperMonitorPage'
import ProcessConsolePage from './pages/ProcessConsolePage'
import RiskCenterPage from './pages/RiskCenterPage'

const nav = [
  { to: '/dashboard', label: '总览' },
  { to: '/offline-training', label: '离线训练' },
  { to: '/paper', label: '模拟盘' },
  { to: '/live', label: '实盘' },
  { to: '/config', label: '配置中心' },
  { to: '/process', label: '进程控制台' },
  { to: '/risk', label: '风控中心' },
]

export default function App() {
  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-100 via-slate-50 to-white">
      <header className="border-b border-slate-200 bg-white/90 backdrop-blur">
        <div className="mx-auto flex max-w-7xl flex-wrap items-center justify-between gap-3 px-4 py-3">
          <div>
            <div className="text-lg font-bold text-slate-900">Quant Control Plane</div>
            <div className="text-xs text-slate-500">strict-only production console</div>
          </div>
          <nav className="flex flex-wrap gap-2">
            {nav.map((n) => (
              <NavLink
                key={n.to}
                to={n.to}
                className={({ isActive }) =>
                  `rounded-lg px-3 py-2 text-sm font-medium ${isActive ? 'bg-slate-900 text-white' : 'bg-slate-100 text-slate-700 hover:bg-slate-200'}`
                }
              >
                {n.label}
              </NavLink>
            ))}
          </nav>
        </div>
      </header>

      <main className="mx-auto max-w-7xl px-4 py-4">
        <Routes>
          <Route path="/" element={<Navigate to="/dashboard" replace />} />
          <Route path="/dashboard" element={<DashboardPage />} />
          <Route path="/offline-training" element={<OfflineTrainingPage />} />
          <Route path="/paper" element={<PaperMonitorPage />} />
          <Route path="/live" element={<LiveMonitorPage />} />
          <Route path="/config" element={<ConfigCenterPage />} />
          <Route path="/process" element={<ProcessConsolePage />} />
          <Route path="/risk" element={<RiskCenterPage />} />
        </Routes>
      </main>
    </div>
  )
}
