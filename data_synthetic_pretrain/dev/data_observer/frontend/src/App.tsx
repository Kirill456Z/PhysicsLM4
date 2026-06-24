import { useEffect } from 'react'
import { api } from './api/client'
import { useStore } from './store'
import TaskViewer from './pages/TaskViewer'

export default function App() {
  const { tabs, selectedTabName, setTabs, setActiveTab } = useStore()

  useEffect(() => {
    // Bootstrap tabs from repo tasks_config.yaml
    api
      .getTaskTabs()
      .then((r) => setTabs(r.tabs))
      .catch(() => {})
  }, [setTabs])

  return (
    <div className="flex flex-col h-full bg-slate-900 text-slate-100">
      <header className="flex items-center gap-6 px-6 py-3 bg-slate-800 border-b border-slate-700 shrink-0">
        <span className="text-sm font-semibold text-emerald-400 tracking-wide">
          Data Observer
        </span>
        <nav className="flex gap-1">
          {(tabs.length ? tabs : [{ tab_name: 'loading', task_name: 'depo', config_yaml: '' }]).map((tab) => (
            <button
              key={tab.tab_name}
              onClick={() => tabs.length && setActiveTab(tab)}
              disabled={!tabs.length}
              className={`px-4 py-1.5 rounded text-xs font-medium uppercase tracking-wide transition-colors ${
                selectedTabName === tab.tab_name
                  ? 'bg-emerald-700 text-white'
                  : 'text-slate-400 hover:text-slate-100 hover:bg-slate-700'
              } ${!tabs.length ? 'opacity-60 cursor-not-allowed' : ''}`}
            >
              {tab.tab_name}
            </button>
          ))}
        </nav>
      </header>

      <main className="flex-1 overflow-hidden">
        <TaskViewer />
      </main>
    </div>
  )
}
