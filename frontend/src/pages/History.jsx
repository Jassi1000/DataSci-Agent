// frontend/src/pages/History.jsx
import { Link } from 'react-router-dom'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Clock, Trash2 } from 'lucide-react'
import { deleteRun, getHistory } from '../api/history'
import ErrorCard from '../components/shared/ErrorCard'
import SkeletonLoader from '../components/shared/SkeletonLoader'

export default function History() {
  const queryClient = useQueryClient()
  const { data: runs = [], isLoading, error } = useQuery({ queryKey: ['history'], queryFn: getHistory, refetchOnWindowFocus: true })
  const mutation = useMutation({ mutationFn: deleteRun, onSuccess: () => queryClient.invalidateQueries({ queryKey: ['history'] }) })
  const remove = (runId) => {
    if (window.confirm('Delete this run history?')) mutation.mutate(runId)
  }
  if (isLoading) return <SkeletonLoader lines={9} height="h-12" />
  if (error) return <ErrorCard title="History unavailable" message="Could not load analysis history." detail={error.message} />
  if (!runs.length) return <div className="rounded-lg bg-surface-700 p-10 text-center"><Clock className="mx-auto h-12 w-12 text-slate-400" /><h2 className="mt-4 text-xl font-semibold text-white">No History Yet</h2><Link to="/dashboard/upload" className="mt-4 inline-flex rounded-lg bg-brand-500 px-4 py-2 font-semibold text-white">Start An Analysis</Link></div>
  return (
    <div className="overflow-hidden rounded-lg bg-surface-700">
      <table className="w-full text-left text-sm">
        <thead className="bg-surface-800 text-slate-300"><tr><th className="p-3">Run ID</th><th className="p-3">Dataset</th><th className="p-3">Date</th><th className="p-3">Problem Type</th><th className="p-3">Best Model</th><th className="p-3">Score</th><th className="p-3">Status</th><th className="p-3">Actions</th></tr></thead>
        <tbody>{runs.map((run) => <tr key={run.run_id} className="border-t border-surface-600"><td className="p-3 text-white">{run.run_id}</td><td className="p-3 text-slate-300">{run.dataset_name}</td><td className="p-3 text-slate-400">{new Date(run.created_at * 1000).toLocaleString()}</td><td className="p-3 text-slate-300">{run.problem_type}</td><td className="p-3 text-slate-300">{run.best_model}</td><td className="p-3 text-slate-300">{typeof run.score === 'number' ? run.score.toFixed(4) : 'N/A'}</td><td className="p-3"><Status status={run.status} /></td><td className="p-3"><div className="flex gap-2"><Link to={`/dashboard/results/${run.run_id}`} className="rounded bg-brand-500 px-3 py-1 text-xs font-semibold text-white">View</Link><button type="button" onClick={() => remove(run.run_id)} className="rounded bg-red-500/20 p-1 text-red-200"><Trash2 className="h-4 w-4" /></button></div></td></tr>)}</tbody>
      </table>
    </div>
  )
}

function Status({ status }) {
  const cls = status === 'completed' ? 'bg-emerald-500/20 text-emerald-200' : status === 'failed' ? 'bg-red-500/20 text-red-200' : 'bg-amber-500/20 text-amber-200'
  return <span className={`rounded px-2 py-1 text-xs ${cls}`}>{status}</span>
}

