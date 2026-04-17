// frontend/src/components/layout/Layout.jsx
import { Outlet, useLocation } from 'react-router-dom'
import Sidebar from './Sidebar'
import TopBar from './TopBar'

export default function Layout() {
  const location = useLocation()
  const isChat = location.pathname.includes('/dashboard/chat/')
  return (
    <div className="light-dashboard min-h-screen bg-slate-50">
      <Sidebar />
      <main className="ml-[260px] flex min-h-screen flex-col">
        <TopBar />
        <section className={`flex-1 overflow-y-auto ${isChat ? 'p-0' : 'p-6'}`}>
          <Outlet />
        </section>
      </main>
    </div>
  )
}
