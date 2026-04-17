// frontend/src/pages/Results.jsx
import { useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { getPipelineResult, getPipelineStatus } from '../api/pipeline'
import CleaningReport from '../components/pipeline/CleaningReport'
import EDAReport from '../components/pipeline/EDAReport'
import EvaluationReport from '../components/pipeline/EvaluationReport'
import ExplainabilityReport from '../components/pipeline/ExplainabilityReport'
import ModelsReport from '../components/pipeline/ModelsReport'
import PlanTimeline from '../components/pipeline/PlanTimeline'
import ProgressTracker from '../components/pipeline/ProgressTracker'
import DownloadPanel from '../components/reports/DownloadPanel'
import ErrorCard from '../components/shared/ErrorCard'
import SkeletonLoader from '../components/shared/SkeletonLoader'

const tabs = ['Plan', 'Cleaning', 'EDA', 'Models', 'Explainability', 'Evaluation', 'Report']

export default function Results() {
  const { runId } = useParams()
  const [activeTab, setActiveTab] = useState('Plan')
  const statusQuery = useQuery({ queryKey: ['status', runId], queryFn: () => getPipelineStatus(runId), enabled: !!runId, refetchInterval: (query) => ['completed', 'failed'].includes(query.state.data?.status) ? false : 1500 })
  const resultQuery = useQuery({ queryKey: ['result', runId], queryFn: () => getPipelineResult(runId), enabled: !!runId && statusQuery.data?.status === 'completed' })
  const status = statusQuery.data?.status
  if (status === 'running' || status === 'not_found' || !status) return <ProgressTracker runId={runId} />
  if (status === 'failed') return <ErrorCard title="Pipeline failed" message="This analysis could not complete." detail={JSON.stringify(statusQuery.data, null, 2)} />
  if (resultQuery.isLoading) return <SkeletonLoader lines={8} height="h-12" />
  if (resultQuery.error) return <ErrorCard title="Result unavailable" message="Could not load run result." detail={resultQuery.error.message} />
  const result = resultQuery.data
  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div><h2 className="text-2xl font-semibold text-slate-900">{result?.dataset_name}</h2><p className="text-sm text-slate-500">{result?.created_at ? new Date(result.created_at * 1000).toLocaleString() : ''}</p></div>
        <div className="flex items-center gap-3"><span className="rounded bg-emerald-100 px-2 py-1 text-xs font-semibold text-emerald-700">{result?.status}</span><Link to={`/dashboard/chat/${runId}`} className="rounded-lg bg-brand-500 px-4 py-2 text-sm font-semibold text-white">Chat With Data</Link></div>
      </div>
      <div className="overflow-hidden rounded-lg bg-white border border-slate-200">
        <div className="flex overflow-x-auto border-b border-slate-200">
          {tabs.map((tab) => <button key={tab} type="button" onClick={() => setActiveTab(tab)} className={`px-5 py-3 text-sm font-medium transition-all duration-200 ${activeTab === tab ? 'border-b-2 border-brand-500 text-slate-900' : 'text-slate-600 hover:text-slate-900 hover:bg-slate-50'}`}>{tab}</button>)}
        </div>
        <div className="p-5 bg-white">
          {activeTab === 'Plan' ? <PlanTimeline planData={result?.plan_results} /> : null}
          {activeTab === 'Cleaning' ? <CleaningReport cleaningData={result?.cleaning_results} /> : null}
          {activeTab === 'EDA' ? <EDAReport edaData={result?.eda_results} /> : null}
          {activeTab === 'Models' ? <ModelsReport mlData={result?.ml_results} /> : null}
          {activeTab === 'Explainability' ? <ExplainabilityReport explainData={result?.explainability_results} /> : null}
          {activeTab === 'Evaluation' ? <EvaluationReport evaluationData={result?.evaluation_results} mlData={result?.ml_results} /> : null}
          {activeTab === 'Report' ? <DownloadPanel runId={runId} /> : null}
        </div>
      </div>
    </div>
  )
}

