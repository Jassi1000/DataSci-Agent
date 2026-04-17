// frontend/src/App.jsx
import { Navigate, Route, Routes } from 'react-router-dom'
import Layout from './components/layout/Layout'
import Chat from './pages/Chat'
import Dashboard from './pages/Dashboard'
import History from './pages/History'
import LandingPage from './pages/LandingPage'
import Results from './pages/Results'
import Upload from './pages/Upload'

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<LandingPage />} />
      <Route path="/dashboard" element={<Layout />}>
        <Route index element={<Dashboard />} />
        <Route path="upload" element={<Upload />} />
        <Route path="results" element={<Navigate to="/dashboard/history" replace />} />
        <Route path="results/:runId" element={<Results />} />
        <Route path="chat" element={<Navigate to="/dashboard/history" replace />} />
        <Route path="chat/:runId" element={<Chat />} />
        <Route path="history" element={<History />} />
      </Route>
    </Routes>
  )
}

