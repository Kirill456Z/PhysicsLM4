import { useMutation } from '@tanstack/react-query'
import { useStore } from '../store'
import { api } from '../api/client'
import ConfigEditor from './ConfigEditor'

export default function SearchPanel() {
  const {
    filterYaml,
    minorVersion,
    setMinorVersion,
    searchResultRuns,
    setSearchResultRuns,
    addToComparison,
    comparisonRuns,
  } = useStore()

  const searchMutation = useMutation({
    mutationFn: () => api.search(filterYaml, minorVersion),
    onSuccess: (data) => setSearchResultRuns(data.runs),
  })

  const comparisonIds = new Set(comparisonRuns.map((r) => r.id))

  return (
    <div className="w-80 shrink-0 flex flex-col border-r border-slate-700 bg-slate-900">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-2 border-b border-slate-700 bg-slate-800 shrink-0">
        <span className="text-xs font-medium text-slate-400 uppercase tracking-wide">Filter</span>
        <div className="flex items-center gap-2">
          <span className="text-xs text-slate-500">minor</span>
          <input
            type="number"
            value={minorVersion < 0 ? '' : minorVersion}
            placeholder="any"
            onChange={(e) =>
              setMinorVersion(e.target.value === '' ? -1 : Number(e.target.value))
            }
            className="w-14 px-1.5 py-0.5 text-xs bg-slate-700 border border-slate-600 rounded text-slate-300 text-center placeholder-slate-600 focus:outline-none focus:border-slate-500"
          />
        </div>
      </div>

      {/* YAML editor */}
      <div className="flex-1 overflow-hidden min-h-0">
        <ConfigEditor />
      </div>

      {/* Search button */}
      <div className="shrink-0 p-3 border-t border-slate-700 space-y-2">
        <button
          onClick={() => searchMutation.mutate()}
          disabled={searchMutation.isPending}
          className="w-full py-2 rounded bg-emerald-700 hover:bg-emerald-600 text-sm font-medium disabled:opacity-50 transition-colors"
        >
          {searchMutation.isPending ? 'Searching…' : 'Search'}
        </button>
        {searchMutation.error && (
          <p className="text-xs text-red-400 break-words">
            {(searchMutation.error as Error).message}
          </p>
        )}
      </div>

      {/* Results */}
      {searchMutation.isSuccess && (
        <div className="shrink-0 border-t border-slate-700 max-h-64 overflow-y-auto">
          <div className="flex items-center justify-between px-4 py-2 sticky top-0 bg-slate-900 border-b border-slate-800">
            <span className="text-xs text-slate-500 uppercase tracking-wide">
              {searchResultRuns.length} run(s) found
            </span>
            {searchResultRuns.length > 0 && (
              <button
                onClick={() => searchResultRuns.forEach((r) => addToComparison(r))}
                className="text-xs px-2 py-0.5 rounded bg-slate-700 hover:bg-slate-600 text-slate-300 transition-colors"
              >
                Add all
              </button>
            )}
          </div>
          {searchResultRuns.map((run) => {
            const added = comparisonIds.has(run.id)
            return (
              <div
                key={run.id}
                className="flex items-center px-3 py-1.5 hover:bg-slate-800 border-b border-slate-800/50"
              >
                <span className="flex-1 text-xs text-slate-300 truncate font-mono" title={run.id}>
                  {run.id}
                </span>
                <button
                  onClick={() => !added && addToComparison(run)}
                  disabled={added}
                  className={`ml-2 w-6 h-6 flex items-center justify-center rounded text-sm font-medium transition-colors ${
                    added
                      ? 'text-emerald-500 cursor-default'
                      : 'text-slate-500 hover:text-emerald-400 hover:bg-slate-700'
                  }`}
                  title={added ? 'Added' : 'Add to comparison'}
                >
                  {added ? '✓' : '+'}
                </button>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}
