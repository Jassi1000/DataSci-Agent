// frontend/src/pages/Chat.jsx
import { useParams } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { getPipelineResult } from '../api/pipeline'
import ChatInterface from '../components/chat/ChatInterface'
import ErrorCard from '../components/shared/ErrorCard'
import SkeletonLoader from '../components/shared/SkeletonLoader'

export default function Chat() {
  const { runId } = useParams()
  const { data, isLoading, error } = useQuery({ queryKey: ['result', runId], queryFn: () => getPipelineResult(runId), enabled: !!runId })
  if (isLoading) return <div className="p-6"><SkeletonLoader lines={8} height="h-12" /></div>
  if (error) return <div className="p-6"><ErrorCard title="Chat unavailable" message="Could not load run metadata." detail={error.message} /></div>
  return <ChatInterface runId={runId} runMeta={data} />
}

