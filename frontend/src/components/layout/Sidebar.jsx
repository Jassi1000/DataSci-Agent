// frontend/src/components/layout/Sidebar.jsx
import { BarChart2, Clock, LayoutDashboard, MessageSquare, Plus } from 'lucide-react'
import { NavLink, Link } from 'react-router-dom'

const items = [
  { to: '/dashboard', label: 'Dashboard', icon: LayoutDashboard, end: true },
  { to: '/dashboard/upload', label: 'New Analysis', icon: Plus },
  { to: '/dashboard/history', label: 'History', icon: Clock },
  { to: '/dashboard/history', label: 'Results', icon: BarChart2 },
  { to: '/dashboard/history', label: 'Chat', icon: MessageSquare },
]

export default function Sidebar() {
  return (
    <aside className="fixed inset-y-0 left-0 z-40 flex w-[260px] flex-col border-r border-surface-700 bg-surface-900">
      <Link to="/dashboard" className="px-6 py-5 text-xl font-bold text-brand-500">
        ⚡ InsightForge
      </Link>
      <nav className="flex-1 space-y-1 px-3">
        {items.map(({ to, label, icon: Icon, end }) => (
          <NavLink
            key={`${label}-${to}`}
            to={to}
            end={end}
            className={({ isActive }) =>
              `flex items-center gap-3 rounded-lg px-3 py-3 text-sm font-medium transition-all duration-200 ${isActive ? 'border-l-2 border-brand-500 bg-surface-700 text-white' : 'text-slate-400 hover:bg-surface-800 hover:text-white group'}`
            }
          >
            <Icon className="h-5 w-5 transition-all duration-200 group-hover:text-indigo-400" />
            {label}
          </NavLink>
        ))}
      </nav>
      <div className="p-4">
        <Link to="/dashboard/upload" className="flex w-full items-center justify-center gap-2 rounded-lg bg-brand-500 px-4 py-3 text-sm font-semibold text-white hover:bg-brand-600">
          <Plus className="h-4 w-4" />
          New Analysis
        </Link>
      </div>
    </aside>
  )
}

