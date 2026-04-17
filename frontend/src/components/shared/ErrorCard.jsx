// frontend/src/components/shared/ErrorCard.jsx
import { useState } from 'react'
import { AlertCircle } from 'lucide-react'

export default function ErrorCard({ title = 'Something went wrong', message = 'The request could not be completed.', detail }) {
  const [showDetail, setShowDetail] = useState(false)
  return (
    <div className="rounded-lg border border-red-500/40 bg-red-500/10 p-5 text-red-100">
      <div className="flex items-start gap-3">
        <AlertCircle className="mt-0.5 h-5 w-5 text-red-400" />
        <div className="min-w-0 flex-1">
          <h3 className="font-semibold text-white">{title}</h3>
          <p className="mt-1 text-sm text-red-100/80">{message}</p>
          {detail ? (
            <div className="mt-3">
              <button type="button" onClick={() => setShowDetail((v) => !v)} className="text-xs font-medium text-red-200 underline">
                Technical Details
              </button>
              {showDetail ? <pre className="mt-3 max-h-64 overflow-auto rounded-lg bg-surface-900 p-3 text-xs text-red-100">{String(detail)}</pre> : null}
            </div>
          ) : null}
        </div>
      </div>
    </div>
  )
}

