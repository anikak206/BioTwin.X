import { NavLink } from 'react-router-dom'
import { Activity, LayoutDashboard, Users, Brain, FileText } from 'lucide-react'
import { cn } from '@/lib/utils'

const nav = [
  { to: '/', label: 'Overview', icon: LayoutDashboard },
  { to: '/triage', label: 'Triage Board', icon: Users },
  { to: '/patients', label: 'Patient Drilldown', icon: Activity },
  { to: '/insights', label: 'AI Insights', icon: Brain },
  { to: '/docs', label: 'Model & Docs', icon: FileText },
]

export default function Sidebar() {
  return (
    <aside className="fixed left-0 top-0 h-screen w-72 border-r border-border bg-surface/90 backdrop-blur">
      <div className="flex h-full flex-col px-6 py-8">
        <div className="mb-10">
          <div className="flex items-center gap-2">
            <div className="h-2 w-2 rounded-full bg-accent shadow-[0_0_40px_rgba(56,189,248,0.8)]" />
            <span className="text-lg font-semibold tracking-tight">BioTwin<span className="text-accent">.X</span></span>
          </div>
          <p className="mt-2 text-sm text-muted">Cross-Modal Contrastive Digital Twin</p>
        </div>
        <nav className="flex flex-col gap-1">
          {nav.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              className={({ isActive }) =>
                cn(
                  'group flex items-center gap-3 rounded-lg px-3 py-2 text-sm transition-colors',
                  isActive ? 'bg-accent/10 text-foreground' : 'text-muted hover:text-foreground hover:bg-white/5'
                )
              }
            >
              <item.icon className="h-4 w-4" />
              {item.label}
            </NavLink>
          ))}
        </nav>
        <div className="mt-auto rounded-xl border border-border bg-panel/60 p-4">
          <p className="text-xs uppercase tracking-[0.3em] text-muted">Status</p>
          <p className="mt-2 text-sm">Research Prototype • Synthetic Data Only</p>
          <p className="mt-1 text-xs text-muted">Not a Medical Device</p>
        </div>
      </div>
    </aside>
  )
}
