import { useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import { api } from '../api/client'
import { useStore } from '../store'
import type { SampleResponse } from '../types'

interface Props {
  sample: SampleResponse
}

function parseTokenInput(raw: string): number[] | null {
  const trimmed = raw.trim()
  if (!trimmed) return null
  // Accept space or comma-separated integers
  const parts = trimmed.split(/[\s,]+/).filter(Boolean)
  const nums = parts.map(Number)
  if (nums.some(isNaN)) return null
  return nums
}

function MetricRow({ name, value }: { name: string; value: number }) {
  const pct = (value * 100).toFixed(1)
  const isGood = value >= 1
  const isMid = value > 0 && value < 1

  return (
    <div className="flex items-center justify-between py-2 px-4 border-b border-slate-800">
      <span className="text-slate-400 font-mono text-xs">{name}</span>
      <div className="flex items-center gap-3">
        <div className="w-32 h-1.5 bg-slate-800 rounded overflow-hidden">
          <div
            className={`h-full rounded transition-all ${isGood ? 'bg-emerald-500' : isMid ? 'bg-yellow-500' : 'bg-red-600'}`}
            style={{ width: `${Math.min(value * 100, 100)}%` }}
          />
        </div>
        <span className={`font-mono text-xs tabular-nums w-16 text-right ${isGood ? 'text-emerald-400' : isMid ? 'text-yellow-400' : 'text-red-400'}`}>
          {Number.isInteger(value) ? value.toFixed(0) : pct + '%'}
        </span>
      </div>
    </div>
  )
}

export default function EvalPanel({ sample }: Props) {
  const { configYaml, selectedTask } = useStore()
  const [input, setInput] = useState('')
  const [parseError, setParseError] = useState<string | null>(null)

  const evalMutation = useMutation({
    mutationFn: (generation: number[]) => {
      const ts = sample.task_specific
      if (ts?.type !== 'depo') throw new Error('Evaluation only supported for depo tasks')

      return api.evaluate({
        config_yaml: configYaml,
        task_name: selectedTask,
        generation,
        task_index: sample.task_index,
        context: sample.context,
        loss_mask: sample.loss_mask,
        answer_start_index: sample.answer_start_index,
        query_nodes: ts.query_nodes,
        answer_nodes: ts.answer_nodes,
        num_hops: ts.num_hops,
        graph_nodes: sample.graph.nodes.map((n) => n.tokens),
        graph_edges: sample.graph.edges,
      })
    },
  })

  function handleEvaluate() {
    setParseError(null)
    const tokens = parseTokenInput(input)
    if (tokens === null) {
      setParseError('Enter space- or comma-separated integers, e.g. "1 5 2 6"')
      return
    }
    evalMutation.mutate(tokens)
  }

  const ts = sample.task_specific
  const answerTokens = ts?.type === 'depo'
    ? ts.answer_nodes.flat().join(' ')
    : null

  return (
    <div className="flex flex-col h-full p-6 gap-6 overflow-auto">
      {/* Context hint */}
      {ts?.type === 'depo' && (
        <section>
          <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-widest mb-3">
            Expected answer tokens
          </h3>
          <p className="text-xs text-slate-500 mb-2">
            The ground-truth continuation starting at position {sample.answer_start_index}:
          </p>
          <div className="flex flex-wrap gap-2">
            {ts.answer_nodes.map((node, i) => (
              <div key={i} className="flex items-center gap-1.5">
                {i > 0 && <span className="text-slate-600 text-xs">·</span>}
                <span className="text-xs text-slate-500">hop {ts.num_hops[i]} answer:</span>
                <span className="font-mono text-xs bg-slate-800 border border-emerald-800 text-emerald-300 rounded px-2 py-0.5">
                  {node.join(' ')}
                </span>
              </div>
            ))}
          </div>
          {answerTokens && (
            <button
              onClick={() => setInput(answerTokens)}
              className="mt-2 text-xs text-slate-500 hover:text-slate-300 underline underline-offset-2"
            >
              paste correct answer
            </button>
          )}
        </section>
      )}

      {/* Input */}
      <section>
        <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-widest mb-3">
          Generation input
        </h3>
        <p className="text-xs text-slate-500 mb-2">
          Token IDs the model would generate (space- or comma-separated):
        </p>
        <div className="flex gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleEvaluate()}
            placeholder="e.g.  1 5  2 6"
            className="flex-1 bg-slate-800 border border-slate-600 rounded px-3 py-2 text-sm font-mono text-slate-200 placeholder-slate-600 focus:outline-none focus:border-emerald-600"
          />
          <button
            onClick={handleEvaluate}
            disabled={evalMutation.isPending}
            className="px-5 py-2 rounded bg-emerald-700 hover:bg-emerald-600 text-sm font-medium disabled:opacity-50 transition-colors"
          >
            {evalMutation.isPending ? 'Running…' : 'Evaluate'}
          </button>
        </div>
        {parseError && <p className="mt-1.5 text-xs text-red-400">{parseError}</p>}
      </section>

      {/* Results */}
      {evalMutation.isError && (
        <section>
          <h3 className="text-xs font-semibold text-red-400 uppercase tracking-widest mb-3">Error</h3>
          <pre className="text-xs text-red-400 bg-red-950/30 border border-red-900 rounded p-3 whitespace-pre-wrap break-words">
            {evalMutation.error?.message}
          </pre>
        </section>
      )}

      {evalMutation.data && (
        <section>
          <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-widest mb-3">
            Metrics
          </h3>
          <div className="border border-slate-700 rounded overflow-hidden">
            {Object.entries(evalMutation.data.metrics).map(([name, value]) => (
              <MetricRow key={name} name={name} value={value as number} />
            ))}
            {Object.keys(evalMutation.data.metrics).length === 0 && (
              <p className="px-4 py-3 text-xs text-slate-500">No metrics returned.</p>
            )}
          </div>
        </section>
      )}
    </div>
  )
}
