import { useMemo, useRef } from 'react'
import CytoscapeComponent from 'react-cytoscapejs'
import type { GraphData, TaskSpecific } from '../types'

interface Props {
  graph: GraphData
  taskSpecific: TaskSpecific
}

const STYLESHEET: cytoscape.StylesheetStyle[] = [
  {
    selector: 'node',
    style: {
      'background-color': '#1e293b',
      'label': 'data(label)',
      'color': '#94a3b8',
      'text-valign': 'center',
      'text-halign': 'center',
      'font-size': '10px',
      'font-family': 'monospace',
      'border-color': '#334155',
      'border-width': 1,
      'width': 64,
      'height': 28,
      'shape': 'round-rectangle',
    },
  },
  {
    selector: 'node.query',
    style: {
      'background-color': '#064e3b',
      'border-color': '#10b981',
      'border-width': 2,
      'color': '#6ee7b7',
    },
  },
  {
    selector: 'node.answer',
    style: {
      'background-color': '#1e3a5f',
      'border-color': '#60a5fa',
      'border-width': 2,
      'color': '#93c5fd',
    },
  },
  {
    selector: 'edge',
    style: {
      'width': 1.5,
      'line-color': '#334155',
      'target-arrow-color': '#475569',
      'target-arrow-shape': 'triangle',
      'source-arrow-color': '#475569',
      'source-arrow-shape': 'none',
      'curve-style': 'bezier',
    },
  },
  {
    selector: 'edge.bidirectional',
    style: {
      'source-arrow-shape': 'triangle',
    },
  },
]

function tokensKey(tokens: number[]): string {
  return JSON.stringify(tokens)
}

export default function GraphViewer({ graph, taskSpecific }: Props) {
  const cyRef = useRef<cytoscape.Core | null>(null)

  const { elements, querySet, answerSet, renderedEdgeCount } = useMemo(() => {
    const querySet = new Set<string>()
    const answerSet = new Set<string>()

    if (taskSpecific?.type === 'depo') {
      taskSpecific.query_nodes.forEach((t) => querySet.add(tokensKey(t)))
      taskSpecific.answer_nodes.forEach((t) => answerSet.add(tokensKey(t)))
    }

    const nodes = graph.nodes.map((node, i) => {
      const key = tokensKey(node.tokens)
      const classes: string[] = []
      if (querySet.has(key)) classes.push('query')
      if (answerSet.has(key)) classes.push('answer')
      return {
        data: { id: String(i), label: `[${node.tokens.join(',')}]` },
        classes: classes.join(' ') || undefined,
      }
    })

    type EdgeElement = {
      data: { id: string; source: string; target: string }
      classes?: string
    }

    const edgeMap = new Map<string, EdgeElement>()

    graph.edges.forEach((edge) => {
      const source = String(edge.from)
      const target = String(edge.to)
      const pairKey =
        edge.from < edge.to ? `${edge.from}|${edge.to}` : `${edge.to}|${edge.from}`
      const existing = edgeMap.get(pairKey)

      if (!existing) {
        edgeMap.set(pairKey, {
          data: { id: `e-${pairKey}`, source, target },
        })
        return
      }

      if (existing.data.source !== source || existing.data.target !== target) {
        existing.classes = 'bidirectional'
      }
    })

    const edges = Array.from(edgeMap.values())

    return {
      elements: [...nodes, ...edges],
      querySet,
      answerSet,
      renderedEdgeCount: edges.length,
    }
  }, [graph, taskSpecific])

  return (
    <div className="relative h-full bg-slate-950">
      <CytoscapeComponent
        key={JSON.stringify(graph)}
        elements={elements}
        stylesheet={STYLESHEET}
        layout={{ name: 'cose', animate: false, padding: 40 } as cytoscape.LayoutOptions}
        style={{ width: '100%', height: '100%' }}
        cy={(cy) => {
          cyRef.current = cy
          cy.one('layoutstop', () => cy.fit(undefined, 40))
        }}
      />

      {/* Legend */}
      <div className="absolute bottom-4 left-4 flex flex-col gap-1 text-xs text-slate-500">
        <span>{graph.n_nodes} nodes &middot; {renderedEdgeCount} edges</span>
        {taskSpecific?.type === 'depo' && (
          <div className="flex gap-3 mt-1">
            <span className="flex items-center gap-1">
              <span className="inline-block w-3 h-3 rounded bg-emerald-900 border border-emerald-500" />
              query
            </span>
            <span className="flex items-center gap-1">
              <span className="inline-block w-3 h-3 rounded bg-blue-900 border border-blue-400" />
              answer
            </span>
          </div>
        )}
      </div>

      {/* Controls */}
      <div className="absolute top-4 right-4 flex gap-2">
        <button
          onClick={() => cyRef.current?.fit(undefined, 40)}
          className="px-2 py-1 text-xs bg-slate-800 hover:bg-slate-700 rounded text-slate-400"
        >
          Fit
        </button>
      </div>
    </div>
  )
}
