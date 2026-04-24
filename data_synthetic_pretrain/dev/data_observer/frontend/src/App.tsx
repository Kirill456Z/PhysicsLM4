import { useEffect } from 'react'
import { useQuery } from '@tanstack/react-query'
import { api } from './api/client'
import { useStore } from './store'
import TaskViewer from './pages/TaskViewer'

export default function App() {
  const { selectedTask, setSelectedTask, setConfigYaml } = useStore()

  const { data: tasksData } = useQuery({
    queryKey: ['tasks'],
    queryFn: api.getTasks,
  })

  const tasks = tasksData?.tasks ?? ['depo']

  useEffect(() => {
    api
      .getDefaultConfig(selectedTask)
      .then((r) => setConfigYaml(r.config_yaml))
      .catch(() => {})
  }, [selectedTask, setConfigYaml])

  return (
    <div className="flex flex-col h-full bg-slate-900 text-slate-100">
      <header className="flex items-center gap-6 px-6 py-3 bg-slate-800 border-b border-slate-700 shrink-0">
        <span className="text-sm font-semibold text-emerald-400 tracking-wide">
          Data Observer
        </span>
        <nav className="flex gap-1">
          {tasks.map((task) => (
            <button
              key={task}
              onClick={() => setSelectedTask(task)}
              className={`px-4 py-1.5 rounded text-xs font-medium uppercase tracking-wide transition-colors ${
                selectedTask === task
                  ? 'bg-emerald-700 text-white'
                  : 'text-slate-400 hover:text-slate-100 hover:bg-slate-700'
              }`}
            >
              {task}
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
