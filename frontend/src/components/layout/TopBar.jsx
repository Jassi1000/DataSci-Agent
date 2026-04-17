// frontend/src/components/layout/TopBar.jsx
import { Bell } from 'lucide-react'
import { useLocation } from 'react-router-dom'

function titleFromPath(pathname) {
  if (pathname.includes('/upload')) return 'New Analysis'
  if (pathname.includes('/results')) return 'Results'
  if (pathname.includes('/chat')) return 'Chat'
  if (pathname.includes('/history')) return 'History'
  return 'Dashboard'
}

export default function TopBar() {
  const location = useLocation()
  return (
    <header className="flex h-16 items-center justify-between border-b border-surface-700 bg-surface-800 px-6">
      <h1 className="text-lg font-semibold text-white">{titleFromPath(location.pathname)}</h1>
      <div className="flex items-center gap-4">
        <button type="button" className="rounded-lg p-2 text-slate-400 transition-all duration-200 hover:bg-surface-700 hover:text-indigo-400"><Bell className="h-5 w-5" /></button>
        <div className="flex h-9 w-9 items-center justify-center rounded-full bg-brand-500 font-semibold text-white">U</div>
      </div>
    </header>
  )
}

