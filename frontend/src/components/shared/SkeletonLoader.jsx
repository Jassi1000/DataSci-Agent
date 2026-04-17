// frontend/src/components/shared/SkeletonLoader.jsx
export default function SkeletonLoader({ lines = 3, height = 'h-4' }) {
  const widths = ['w-full', 'w-5/6', 'w-2/3', 'w-3/4', 'w-1/2']
  return (
    <div className="space-y-3">
      {Array.from({ length: lines }).map((_, i) => (
        <div key={i} className={`${height} ${widths[i % widths.length]} animate-pulse rounded bg-surface-600`} />
      ))}
    </div>
  )
}

