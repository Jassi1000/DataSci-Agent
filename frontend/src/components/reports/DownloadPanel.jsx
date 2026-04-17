// frontend/src/components/reports/DownloadPanel.jsx
import { useState } from 'react'
import { Download, FileText, Loader2, Table } from 'lucide-react'

const downloads = [
  { type: 'pdf', title: 'PDF Report', desc: 'Executive report with metrics and recommendations.', icon: FileText, color: 'text-rose-400' },
  { type: 'excel', title: 'Excel Data', desc: 'Workbook with model, SHAP, and cleaning sheets.', icon: Table, color: 'text-emerald-400' },
  { type: 'csv', title: 'CSV Metrics', desc: 'Flat summary of model comparison metrics.', icon: Download, color: 'text-brand-400' },
]

export default function DownloadPanel({ runId }) {
  const [loading, setLoading] = useState(null)
  const trigger = (type) => {
    setLoading(type)
    window.location.href = `/api/export/${type}/${runId}`
    setTimeout(() => setLoading(null), 700)
  }
  return (
    <div className="grid gap-4 md:grid-cols-3">
      {downloads.map(({ type, title, desc, icon: Icon, color }) => (
        <div key={type} className="rounded-lg bg-surface-700 p-5">
          <Icon className={`h-8 w-8 ${color}`} />
          <h3 className="mt-4 font-semibold text-white">{title}</h3>
          <p className="mt-2 text-sm text-slate-400">{desc}</p>
          <button type="button" onClick={() => trigger(type)} className="mt-5 flex items-center gap-2 rounded-lg bg-brand-500 px-4 py-2 text-sm font-semibold text-white hover:bg-brand-600">
            {loading === type ? <Loader2 className="h-4 w-4 animate-spin" /> : <Download className="h-4 w-4" />}
            Generate & Download
          </button>
        </div>
      ))}
    </div>
  )
}

