// frontend/src/components/pipeline/ProgressTracker.jsx
import { useEffect, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Loader2 } from 'lucide-react'
import { useNavigate } from 'react-router-dom'
import { getPipelineStatus } from '../../api/pipeline'
import ErrorCard from '../shared/ErrorCard'

const agents = [
  ['planner', 'Analysis Planning'],
  ['data_cleaner', 'Data Cleaning'],
  ['eda', 'Exploratory Analysis'],
  ['ml_trainer', 'Model Training'],
  ['explainer', 'Explainability'],
  ['evaluator', 'Evaluation'],
  ['reporter', 'Report Generation'],
]

export default function ProgressTracker({ runId }) {
  const navigate = useNavigate()
  const [elapsed, setElapsed] = useState(0)
  const { data, error } = useQuery({
    queryKey: ['pipeline-status', runId],
    queryFn: () => getPipelineStatus(runId),
    enabled: !!runId,
    refetchInterval: (query) => {
      const status = query.state.data?.status
      return status === 'completed' || status === 'failed' ? false : 1500
    },
  })

  useEffect(() => {
    const t = setInterval(() => setElapsed((v) => v + 1), 1000)
    return () => clearInterval(t)
  }, [])

  useEffect(() => {
    if (data?.status === 'completed') {
      const t = setTimeout(() => navigate(`/dashboard/results/${runId}`), 1500)
      return () => clearTimeout(t)
    }
  }, [data?.status, navigate, runId])

  if (error) return <ErrorCard title="Progress unavailable" message="Could not load pipeline progress." detail={error.message} />
  if (data?.status === 'failed') return <ErrorCard title="Pipeline failed" message={data?.error || 'The pipeline stopped before completion.'} detail={JSON.stringify(data?.logs || [], null, 2)} />

  const pct = data?.progress_pct || 0
  const completed = new Set(data?.completed_agents || [])
  const current = data?.current_agent
  const latest = [...(data?.logs || [])].reverse().find((log) => log.agent === current)?.message

  return (
    <div className="mx-auto max-w-4xl rounded-lg bg-surface-700 p-6">
      <div className="mb-6 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Loader2 className="h-5 w-5 animate-spin text-brand-400" />
          <h2 className="text-xl font-semibold text-white">{data?.status === 'completed' ? 'Pipeline Complete' : 'Pipeline Running...'}</h2>
        </div>
        <span className="text-sm text-slate-400">Elapsed {data?.elapsed_seconds ?? elapsed}s</span>
      </div>
      <div className="mb-2 h-3 overflow-hidden rounded-full bg-surface-600">
        <div className="h-full bg-brand-500 transition-all duration-500" style={{ width: `${pct}%` }} />
      </div>
      <p className="mb-6 text-right text-sm text-slate-300">{pct}% Complete</p>
      <div className="space-y-3">
        {agents.map(([key, label]) => {
          const isDone = completed.has(key)
          const isCurrent = current === key && !isDone
          return (
            <div key={key} className="flex items-center gap-3 rounded-lg bg-surface-800 p-3">
              <span className={`h-3 w-3 rounded-full ${isDone ? 'bg-emerald-400' : isCurrent ? 'animate-pulse bg-brand-400' : 'bg-surface-500'}`} />
              <div>
                <p className="font-medium text-white">{label}</p>
                {isCurrent && latest ? <p className="text-sm text-slate-400">{latest}</p> : null}
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}

