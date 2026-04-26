import type { SampleResponse } from '../types'

interface Props {
  sample: SampleResponse
}

interface TokenInfo {
  type: string
  colorClass: string
}

function interpretToken(token: number, B: number): TokenInfo {
  if (token === -100) return { type: 'NO_LABEL', colorClass: 'text-slate-600' }
  if (token === 0) return { type: 'PAD', colorClass: 'text-slate-600' }
  if (token >= 1 && token <= B) return { type: 'BASE', colorClass: 'text-blue-300' }
  if (token >= B + 1 && token <= 2 * B) return { type: 'EOW', colorClass: 'text-blue-500' }
  if (token === 2 * B + 1) return { type: 'TASK_DEPO', colorClass: 'text-purple-400' }
  if (token === 2 * B + 2) return { type: 'TASK_BREVO', colorClass: 'text-purple-400' }
  if (token === 2 * B + 3) return { type: 'TASK_CONCOMP', colorClass: 'text-purple-400' }
  if (token === 2 * B + 4) return { type: 'EOS', colorClass: 'text-red-500' }
  if (token === 2 * B + 5) return { type: 'BREVO_QUERY', colorClass: 'text-yellow-400' }
  if (token === 2 * B + 6) return { type: 'BREVO_ANS', colorClass: 'text-yellow-400' }
  if (token === 2 * B + 7) return { type: 'CONCOMP_QUERY', colorClass: 'text-yellow-400' }
  if (token === 2 * B + 8) return { type: 'CONCOMP_ANS', colorClass: 'text-yellow-400' }
  if (token === 2 * B + 9) return { type: 'DEPO_SEP', colorClass: 'text-yellow-400' }
  if (token === 2 * B + 10) return { type: 'DEPO_EDGE_SEP', colorClass: 'text-yellow-400' }
  if (token === 2 * B + 11) return { type: 'TASK_SP', colorClass: 'text-purple-400' }
  if (token === 2 * B + 12) return { type: 'SP_QUERY', colorClass: 'text-yellow-400' }
  if (token === 2 * B + 13) return { type: 'SP_ANS', colorClass: 'text-yellow-400' }
  if (token === 2 * B + 14) return { type: 'TASK_CCF', colorClass: 'text-purple-400' }
  if (token >= 200 && token <= 216) return { type: `HOP_${token - 200}`, colorClass: 'text-orange-400' }
  return { type: 'tok', colorClass: 'text-slate-300' }
}

export default function DataInspector({ sample }: Props) {
  const B = sample.base_vocab_size
  const tokens = sample.context_padded
  const labels = sample.labels
  const { answer_start_index: answerAt } = sample

  const labelCount = labels.filter((l) => l !== -100).length
  const padCount = tokens.filter((t) => t === 0).length

  return (
    <div className="flex flex-col h-full text-xs">
      {/* Stats + legend */}
      <div className="flex items-center gap-6 px-4 py-2 border-b border-slate-700 bg-slate-800 shrink-0">
        <div className="flex gap-4 text-slate-500">
          <span>len={tokens.length}</span>
          <span className="text-emerald-600">labels×{labelCount}</span>
          <span className="text-slate-600">pad×{padCount}</span>
        </div>
        <div className="flex gap-3 ml-auto text-slate-500">
          <span className="text-blue-300">BASE</span>
          <span className="text-blue-500">EOW</span>
          <span className="text-purple-400">TASK</span>
          <span className="text-yellow-400">SEP</span>
          <span className="text-orange-400">HOP_k</span>
          <span className="bg-emerald-950/60 px-1 rounded text-emerald-400">label≠-100</span>
        </div>
      </div>

      {/* Table */}
      <div className="flex-1 overflow-auto">
        <table className="w-full border-collapse">
          <thead className="sticky top-0 bg-slate-900 z-10">
            <tr className="text-slate-500 text-left">
              <th className="px-3 py-1.5 font-normal">#</th>
              <th className="px-3 py-1.5 font-normal">is_eval_context</th>
              <th className="px-3 py-1.5 font-normal">token</th>
              <th className="px-3 py-1.5 font-normal">type</th>
              <th className="px-3 py-1.5 font-normal">label</th>
              <th className="px-3 py-1.5 font-normal">label type</th>
            </tr>
          </thead>
          <tbody>
            {tokens.map((tok, i) => {
              const label = labels[i]
              const hasLabel = label !== -100
              const tokInfo = interpretToken(tok, B)
              const lblInfo = interpretToken(label, B)
              const isEvalContext = i < answerAt ? 1 : 0

              return (
                <tr key={i} className={hasLabel ? 'bg-emerald-950/40' : ''}>
                  <td className="px-3 py-0.5 text-slate-600 tabular-nums select-none">{i}</td>
                  <td className="px-3 py-0.5 font-mono tabular-nums text-slate-500">{isEvalContext}</td>
                  <td className={`px-3 py-0.5 font-mono tabular-nums ${tokInfo.colorClass}`}>{tok}</td>
                  <td className="px-3 py-0.5 text-slate-500">{tokInfo.type}</td>
                  <td className={`px-3 py-0.5 font-mono tabular-nums ${hasLabel ? lblInfo.colorClass : 'text-slate-700'}`}>
                    {label}
                  </td>
                  <td className={`px-3 py-0.5 ${hasLabel ? 'text-slate-500' : 'text-slate-800'}`}>
                    {hasLabel ? lblInfo.type : ''}
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </div>
  )
}
