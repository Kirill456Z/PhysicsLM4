import SearchPanel from './components/SearchPanel'
import ComparisonPanel from './components/ComparisonPanel'

export default function App() {
  return (
    <div className="flex flex-col h-full bg-slate-900 text-slate-100">
      <header className="flex items-center gap-4 px-6 py-3 bg-slate-800 border-b border-slate-700 shrink-0">
        <span className="text-sm font-semibold text-emerald-400 tracking-wide">Run Comparator</span>
        <span className="text-xs text-slate-500">recipe_stashes/exps</span>
      </header>
      <main className="flex flex-1 overflow-hidden">
        <SearchPanel />
        <ComparisonPanel />
      </main>
    </div>
  )
}
