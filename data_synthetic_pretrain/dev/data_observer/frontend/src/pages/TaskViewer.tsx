import ConfigEditor from '../components/ConfigEditor'
import GraphViewer from '../components/GraphViewer'
import DataInspector from '../components/DataInspector'
import TaskMetadata from '../components/TaskMetadata'
import { useStore } from '../store'
import { useGenerateSample, useGenerateBatch, useValidateConfig } from '../hooks/useDataGeneration'

const RESULT_TABS = ['graph', 'tokens', 'metadata'] as const
type ResultTab = (typeof RESULT_TABS)[number]

export default function TaskViewer() {
  const {
    currentSample,
    batchSamples,
    batchIndex,
    setBatchIndex,
    activeResultTab,
    setActiveResultTab,
  } = useStore()

  const generateSample = useGenerateSample()
  const generateBatch = useGenerateBatch(5)
  const validate = useValidateConfig()

  const isLoading = generateSample.isPending || generateBatch.isPending
  const error = generateSample.error?.message ?? generateBatch.error?.message ?? null

  return (
    <div className="flex h-full">
      {/* Left: Config Panel */}
      <div className="w-96 shrink-0 flex flex-col border-r border-slate-700 bg-slate-900">
        {/* Config header */}
        <div className="flex items-center justify-between px-4 py-2 border-b border-slate-700 shrink-0">
          <span className="text-xs font-medium text-slate-400 uppercase tracking-wide">Config</span>
          <button
            onClick={() => validate.mutate()}
            disabled={isLoading}
            className="px-3 py-1 text-xs rounded bg-slate-700 hover:bg-slate-600 text-slate-300 disabled:opacity-50 transition-colors"
          >
            Validate
          </button>
        </div>

        {/* Validation result */}
        {validate.data && (
          <div
            className={`px-4 py-1.5 text-xs border-b shrink-0 ${
              validate.data.valid
                ? 'border-emerald-800 text-emerald-400 bg-emerald-950/40'
                : 'border-red-800 text-red-400 bg-red-950/40'
            }`}
          >
            {validate.data.valid
              ? `✓ valid  [${validate.data.parsed?.tasks?.join(', ')}]`
              : validate.data.errors.join(' · ')}
          </div>
        )}

        {/* Editor */}
        <div className="flex-1 overflow-hidden">
          <ConfigEditor />
        </div>

        {/* Actions */}
        <div className="shrink-0 p-3 border-t border-slate-700 space-y-2">
          <div className="flex gap-2">
            <button
              onClick={() => generateSample.mutate()}
              disabled={isLoading}
              className="flex-1 py-2 rounded bg-emerald-700 hover:bg-emerald-600 text-sm font-medium disabled:opacity-50 transition-colors"
            >
              {generateSample.isPending ? 'Generating…' : 'Generate Sample'}
            </button>
            <button
              onClick={() => generateBatch.mutate()}
              disabled={isLoading}
              title="Generate batch of 5"
              className="px-4 py-2 rounded bg-slate-700 hover:bg-slate-600 text-sm disabled:opacity-50 transition-colors text-slate-300"
            >
              ×5
            </button>
          </div>

          {error && <p className="text-xs text-red-400 break-words">{error}</p>}
        </div>
      </div>

      {/* Right: Result Panel */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Result tabs + batch selector */}
        <div className="flex items-center gap-1 px-4 py-2 border-b border-slate-700 bg-slate-800 shrink-0">
          {RESULT_TABS.map((tab) => (
            <button
              key={tab}
              onClick={() => setActiveResultTab(tab as ResultTab)}
              className={`px-4 py-1.5 rounded text-xs capitalize transition-colors ${
                activeResultTab === tab
                  ? 'bg-slate-600 text-white'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-700'
              }`}
            >
              {tab}
            </button>
          ))}

          {batchSamples.length > 1 && (
            <div className="ml-auto flex items-center gap-1.5 text-xs text-slate-400">
              <span className="mr-1">batch:</span>
              {batchSamples.map((_, i) => (
                <button
                  key={i}
                  onClick={() => setBatchIndex(i)}
                  className={`w-6 h-6 rounded text-xs transition-colors ${
                    i === batchIndex
                      ? 'bg-emerald-700 text-white'
                      : 'bg-slate-700 hover:bg-slate-600 text-slate-400'
                  }`}
                >
                  {i + 1}
                </button>
              ))}
            </div>
          )}
        </div>

        {/* Result content */}
        <div className="flex-1 overflow-hidden">
          {!currentSample && !isLoading && (
            <div className="flex flex-col items-center justify-center h-full text-slate-600 gap-2">
              <svg
                className="w-12 h-12 opacity-30"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={1}
                  d="M9 17V7m0 10a2 2 0 01-2 2H5a2 2 0 01-2-2V7a2 2 0 012-2h2a2 2 0 012 2m0 10a2 2 0 002 2h2a2 2 0 002-2M9 7a2 2 0 012-2h2a2 2 0 012 2m0 10V7"
                />
              </svg>
              <p className="text-sm">Press Generate Sample to begin</p>
            </div>
          )}

          {isLoading && (
            <div className="flex items-center justify-center h-full text-slate-400 text-sm gap-2">
              <svg className="animate-spin w-4 h-4" viewBox="0 0 24 24" fill="none">
                <circle
                  className="opacity-25"
                  cx="12"
                  cy="12"
                  r="10"
                  stroke="currentColor"
                  strokeWidth="4"
                />
                <path
                  className="opacity-75"
                  fill="currentColor"
                  d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
                />
              </svg>
              Generating…
            </div>
          )}

          {currentSample && !isLoading && (
            <>
              {activeResultTab === 'graph' && (
                <GraphViewer
                  sample={currentSample}
                  graph={currentSample.graph}
                  taskSpecific={currentSample.task_specific}
                />
              )}
              {activeResultTab === 'tokens' && <DataInspector sample={currentSample} />}
              {activeResultTab === 'metadata' && <TaskMetadata sample={currentSample} />}
            </>
          )}
        </div>
      </div>
    </div>
  )
}
