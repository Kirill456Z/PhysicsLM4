import type { SampleResponse } from '../types'

interface Props {
  sample: SampleResponse
}

function TokenBadge({ tokens }: { tokens: number[] }) {
  return (
    <span className="inline-flex items-center gap-0.5 bg-slate-800 border border-slate-600 rounded px-2 py-0.5 font-mono text-xs text-slate-300">
      [{tokens.join(', ')}]
    </span>
  )
}

function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <tr className="border-b border-slate-800">
      <td className="py-2 pr-6 text-slate-500 text-xs whitespace-nowrap align-top">{label}</td>
      <td className="py-2 text-slate-200 text-xs align-top">{children}</td>
    </tr>
  )
}

function DepoMetadata({ sample }: { sample: SampleResponse }) {
  const ts = sample.task_specific
  if (ts?.type !== 'depo') return null

  return (
    <div className="space-y-6">
      <section>
        <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-widest mb-3">
          Task Info
        </h3>
        <table className="w-full">
          <tbody>
            <Row label="task_name">{sample.task_name}</Row>
            <Row label="task_index">{sample.task_index}</Row>
            <Row label="base_vocab_size">{sample.base_vocab_size}</Row>
            <Row label="context_len">{sample.context.length} tokens</Row>
            <Row label="loss_count">
              {sample.loss_mask.filter((x) => x === 1).length} positions
            </Row>
          </tbody>
        </table>
      </section>

      <section>
        <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-widest mb-3">
          Graph
        </h3>
        <table className="w-full">
          <tbody>
            <Row label="n_nodes">{sample.graph.n_nodes}</Row>
            <Row label="n_edges">{sample.graph.edges.length}</Row>
            <Row label="nodes">
              <div className="flex flex-wrap gap-1">
                {sample.graph.nodes.map((n, i) => (
                  <TokenBadge key={i} tokens={n.tokens} />
                ))}
              </div>
            </Row>
          </tbody>
        </table>
      </section>

      <section>
        <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-widest mb-3">
          Queries ({ts.query_nodes.length})
        </h3>
        <div className="space-y-3">
          {ts.query_nodes.map((qn, i) => (
            <div key={i} className="flex items-center gap-3 bg-slate-800/60 rounded px-3 py-2">
              <span className="text-slate-500 text-xs w-5 shrink-0">#{i}</span>
              <div className="flex items-center gap-2 flex-wrap">
                <span className="text-slate-500 text-xs">query</span>
                <TokenBadge tokens={qn} />
                <span className="text-slate-500 text-xs">
                  +{ts.num_hops[i]} hop{ts.num_hops[i] !== 1 ? 's' : ''}
                </span>
                <span className="text-slate-500 text-xs">→</span>
                <span className="text-slate-500 text-xs">answer</span>
                <TokenBadge tokens={ts.answer_nodes[i]} />
              </div>
            </div>
          ))}
        </div>
      </section>
    </div>
  )
}

export default function TaskMetadata({ sample }: Props) {
  return (
    <div className="h-full overflow-auto p-6">
      {sample.task_specific?.type === 'depo' && <DepoMetadata sample={sample} />}

      {!sample.task_specific && (
        <div className="space-y-6">
          <section>
            <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-widest mb-3">
              Task Info
            </h3>
            <table className="w-full">
              <tbody>
                <Row label="task_name">{sample.task_name}</Row>
                <Row label="task_index">{sample.task_index}</Row>
                <Row label="context_len">{sample.context.length} tokens</Row>
              </tbody>
            </table>
          </section>

          <section>
            <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-widest mb-3">
              Graph
            </h3>
            <table className="w-full">
              <tbody>
                <Row label="n_nodes">{sample.graph.n_nodes}</Row>
                <Row label="n_edges">{sample.graph.edges.length}</Row>
              </tbody>
            </table>
          </section>
        </div>
      )}
    </div>
  )
}
