import { useEffect, useMemo, useState } from 'react'
import { useStore } from '../store'
import type { RunInfo } from '../types'

// ── helpers ──────────────────────────────────────────────────────────────────

function unflatten(flat: Record<string, string>): Record<string, unknown> {
  const result: Record<string, unknown> = {}
  for (const [key, value] of Object.entries(flat)) {
    const parts = key.split('.')
    let node = result as Record<string, unknown>
    for (let i = 0; i < parts.length - 1; i++) {
      if (typeof node[parts[i]] !== 'object' || node[parts[i]] === null) {
        node[parts[i]] = {}
      }
      node = node[parts[i]] as Record<string, unknown>
    }
    node[parts[parts.length - 1]] = value
  }
  return result
}

function toYaml(obj: unknown, indent = 0): string {
  if (typeof obj !== 'object' || obj === null) return String(obj)
  const pad = '  '.repeat(indent)
  return Object.entries(obj as Record<string, unknown>)
    .map(([k, v]) =>
      typeof v === 'object' && v !== null
        ? `${pad}${k}:\n${toYaml(v, indent + 1)}`
        : `${pad}${k}: ${v}`
    )
    .join('\n')
}

function toPythonDict(obj: Record<string, Record<string, string[]>>): string {
  const indent = '    '
  const lines = ['{']
  for (const [col, rows] of Object.entries(obj)) {
    lines.push(`${indent}'${col}': {`)
    for (const [row, runs] of Object.entries(rows)) {
      const items = runs.map((r) => `'${r}'`).join(', ')
      lines.push(`${indent}${indent}'${row}': [${items}],`)
    }
    lines.push(`${indent}},`)
  }
  lines.push('}')
  return lines.join('\n')
}

function useCopy() {
  const [copied, setCopied] = useState<string | null>(null)
  const copy = (text: string, label: string) => {
    navigator.clipboard.writeText(text).then(() => {
      setCopied(label)
      setTimeout(() => setCopied(null), 1500)
    })
  }
  return { copied, copy }
}

// ── Config detail pane ───────────────────────────────────────────────────────

function ConfigDetail({ run, onClose }: { run: RunInfo; onClose: () => void }) {
  const { copied, copy } = useCopy()

  const groups = useMemo(() => {
    const map: Record<string, [string, string][]> = {}
    for (const [k, v] of Object.entries(run.config).sort()) {
      const section = k.includes('.') ? k.split('.')[0] : '—'
      ;(map[section] ??= []).push([k, v])
    }
    return Object.entries(map).sort()
  }, [run.config])

  const yamlText = useMemo(() => toYaml(unflatten(run.config)), [run.config])

  return (
    <div className="w-80 shrink-0 flex flex-col border-l border-slate-700 bg-slate-900">
      <div className="flex items-center gap-2 px-3 py-2 bg-slate-800 border-b border-slate-700 shrink-0">
        <span className="flex-1 text-xs font-mono text-slate-300 truncate min-w-0" title={run.id}>{run.id}</span>
        <button
          onClick={() => copy(run.id, 'name')}
          className="shrink-0 text-xs px-1.5 py-0.5 rounded bg-slate-700 hover:bg-slate-600 text-slate-400 hover:text-slate-200 transition-colors"
        >
          {copied === 'name' ? '✓ name' : 'copy name'}
        </button>
        <button
          onClick={() => copy(yamlText, 'yaml')}
          className="shrink-0 text-xs px-1.5 py-0.5 rounded bg-slate-700 hover:bg-slate-600 text-slate-400 hover:text-slate-200 transition-colors"
        >
          {copied === 'yaml' ? '✓ yaml' : 'copy yaml'}
        </button>
        <button onClick={onClose} className="shrink-0 text-slate-500 hover:text-slate-300 text-lg leading-none">
          ×
        </button>
      </div>
      <div className="flex-1 overflow-y-auto">
        {groups.map(([section, pairs]) => (
          <div key={section}>
            <div className="px-4 py-1.5 text-xs font-medium text-emerald-500 uppercase tracking-wide bg-slate-800/60 border-b border-slate-800 sticky top-0">
              {section}
            </div>
            {pairs.map(([k, v]) => {
              const shortKey = k.includes('.') ? k.slice(k.indexOf('.') + 1) : k
              return (
                <div key={k} className="flex gap-2 px-4 py-1 border-b border-slate-800/50 hover:bg-slate-800/30">
                  <span className="text-xs font-mono text-slate-500 shrink-0 w-36 truncate" title={k}>{shortKey}</span>
                  <span className="text-xs font-mono text-slate-300 break-all min-w-0">{v}</span>
                </div>
              )
            })}
          </div>
        ))}
      </div>
    </div>
  )
}

// ── Main panel ────────────────────────────────────────────────────────────────

export default function ComparisonPanel() {
  const { comparisonRuns, removeFromComparison } = useStore()
  const [param1, setParam1] = useState('')
  const [param2, setParam2] = useState('')
  const [selectedRun, setSelectedRun] = useState<RunInfo | null>(null)
  const [selectedRows, setSelectedRows] = useState<Set<string>>(new Set())
  const [selectedCols, setSelectedCols] = useState<Set<string>>(new Set())
  const [rowsFirst, setRowsFirst] = useState(false)
  const { copied, copy } = useCopy()

  const { differingKeys, binTable, rowVals, colVals } = useMemo(() => {
    const empty = {
      differingKeys: [] as string[],
      binTable: null as Record<string, Record<string, string[]>> | null,
      rowVals: [] as string[],
      colVals: [] as string[],
    }
    if (comparisonRuns.length < 2) return empty

    const allKeys = new Set<string>()
    comparisonRuns.forEach((r) => Object.keys(r.config).forEach((k) => allKeys.add(k)))

    const differingKeys = [...allKeys].sort().filter((key) => {
      const vals = new Set(comparisonRuns.map((r) => r.config[key] ?? '__missing__'))
      return vals.size > 1
    })

    if (!param1 || !param2 || param1 === param2) return { ...empty, differingKeys }

    const rowVals = [...new Set(comparisonRuns.map((r) => r.config[param1] ?? 'N/A'))].sort()
    const colVals = [...new Set(comparisonRuns.map((r) => r.config[param2] ?? 'N/A'))].sort()

    const binTable: Record<string, Record<string, string[]>> = {}
    rowVals.forEach((rv) => {
      binTable[rv] = {}
      colVals.forEach((cv) => { binTable[rv][cv] = [] })
    })
    comparisonRuns.forEach((r) => {
      const rv = r.config[param1] ?? 'N/A'
      const cv = r.config[param2] ?? 'N/A'
      binTable[rv][cv].push(r.id)
    })

    return { differingKeys, binTable, rowVals, colVals }
  }, [comparisonRuns, param1, param2])

  // Reset row/col selections whenever the param axes change
  useEffect(() => {
    setSelectedRows(new Set(rowVals))
    setSelectedCols(new Set(colVals))
  }, [param1, param2, rowVals.join('|'), colVals.join('|')])

  const toggleRow = (rv: string) =>
    setSelectedRows((prev) => { const s = new Set(prev); s.has(rv) ? s.delete(rv) : s.add(rv); return s })
  const toggleCol = (cv: string) =>
    setSelectedCols((prev) => { const s = new Set(prev); s.has(cv) ? s.delete(cv) : s.add(cv); return s })

  const visibleRows = rowVals.filter((rv) => selectedRows.has(rv))
  const visibleCols = colVals.filter((cv) => selectedCols.has(cv))

  const handleExtract = () => {
    if (!binTable) return
    const out: Record<string, Record<string, string[]>> = {}
    if (rowsFirst) {
      for (const rv of visibleRows) {
        out[rv] = {}
        for (const cv of visibleCols) out[rv][cv] = binTable[rv][cv]
      }
    } else {
      for (const cv of visibleCols) {
        out[cv] = {}
        for (const rv of visibleRows) out[cv][rv] = binTable[rv][cv]
      }
    }
    copy(toPythonDict(out), 'extract')
  }

  const resolvedSelected = selectedRun && comparisonRuns.some((r) => r.id === selectedRun.id)
    ? selectedRun : null

  if (comparisonRuns.length === 0) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center text-slate-600 gap-2">
        <svg className="w-10 h-10 opacity-30" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1}
            d="M9 17V7m0 10a2 2 0 01-2 2H5a2 2 0 01-2-2V7a2 2 0 012-2h2a2 2 0 012 2m0 10a2 2 0 002 2h2a2 2 0 002-2M9 7a2 2 0 012-2h2a2 2 0 012 2m0 10V7" />
        </svg>
        <p className="text-sm">Search for runs and add them with +</p>
      </div>
    )
  }

  return (
    <div className="flex-1 flex overflow-hidden min-w-0">
      {/* ── Left: comparison ── */}
      <div className="flex-1 flex flex-col overflow-hidden min-w-0">
        <div className="px-4 py-2 border-b border-slate-700 bg-slate-800 shrink-0">
          <span className="text-xs font-medium text-slate-400 uppercase tracking-wide">Compare</span>
        </div>

        {/* Runs list */}
        <div className="shrink-0 px-4 py-3 border-b border-slate-700 max-h-44 overflow-y-auto">
          <p className="text-xs text-slate-500 uppercase tracking-wide mb-2">Runs ({comparisonRuns.length})</p>
          <div className="space-y-1">
            {comparisonRuns.map((run) => (
              <div key={run.id} className="flex items-center gap-2 px-2 py-1 rounded bg-slate-800 group">
                <button
                  onClick={() => setSelectedRun(resolvedSelected?.id === run.id ? null : run)}
                  className={`flex-1 text-xs font-mono text-left truncate transition-colors ${
                    resolvedSelected?.id === run.id ? 'text-emerald-400' : 'text-slate-300 hover:text-emerald-400'
                  }`}
                  title={`${run.id} — click to view config`}
                >
                  {run.id}
                </button>
                <button
                  onClick={() => { if (resolvedSelected?.id === run.id) setSelectedRun(null); removeFromComparison(run.id) }}
                  className="text-slate-600 hover:text-red-400 text-base leading-none transition-colors shrink-0"
                >×</button>
              </div>
            ))}
          </div>
        </div>

        {comparisonRuns.length < 2 ? (
          <div className="flex-1 flex items-center justify-center text-slate-600 text-sm">
            Add at least 2 runs to compare
          </div>
        ) : (
          <>
            {/* Differing params */}
            <div className="shrink-0 px-4 py-3 border-b border-slate-700">
              <p className="text-xs text-slate-500 uppercase tracking-wide mb-2">
                Differing parameters ({differingKeys.length})
              </p>
              <div className="flex flex-wrap gap-1">
                {differingKeys.map((k) => (
                  <span
                    key={k}
                    className="px-1.5 py-0.5 text-xs bg-slate-700 text-slate-300 rounded font-mono cursor-pointer hover:bg-slate-600 transition-colors"
                    onClick={() => { if (!param1) setParam1(k); else if (!param2 && k !== param1) setParam2(k) }}
                    title="Click to select as row/column"
                  >
                    {k}
                  </span>
                ))}
              </div>
            </div>

            {/* Param selectors */}
            <div className="shrink-0 px-4 py-3 border-b border-slate-700 flex gap-4 items-end">
              <div className="flex-1">
                <label className="text-xs text-slate-500 uppercase tracking-wide block mb-1">Row</label>
                <select value={param1} onChange={(e) => setParam1(e.target.value)}
                  className="w-full px-2 py-1 text-xs bg-slate-700 border border-slate-600 rounded text-slate-300 focus:outline-none focus:border-slate-500">
                  <option value="">—</option>
                  {differingKeys.map((k) => <option key={k} value={k}>{k}</option>)}
                </select>
              </div>
              <div className="flex-1">
                <label className="text-xs text-slate-500 uppercase tracking-wide block mb-1">Column</label>
                <select value={param2} onChange={(e) => setParam2(e.target.value)}
                  className="w-full px-2 py-1 text-xs bg-slate-700 border border-slate-600 rounded text-slate-300 focus:outline-none focus:border-slate-500">
                  <option value="">—</option>
                  {differingKeys.map((k) => <option key={k} value={k}>{k}</option>)}
                </select>
              </div>
              {(param1 || param2) && (
                <button onClick={() => { setParam1(''); setParam2('') }}
                  className="text-xs text-slate-500 hover:text-slate-300 transition-colors pb-1">
                  clear
                </button>
              )}
            </div>

            {binTable ? (
              <>
                {/* Extract button + direction toggle */}
                <div className="shrink-0 px-4 py-2 border-b border-slate-700 flex items-center justify-end gap-3">
                  <div className="flex items-center gap-1 text-xs text-slate-500">
                    <span className={rowsFirst ? 'text-slate-500' : 'text-slate-300'}>cols first</span>
                    <button
                      onClick={() => setRowsFirst((v) => !v)}
                      className={`relative w-9 h-5 rounded-full transition-colors ${rowsFirst ? 'bg-emerald-700' : 'bg-slate-600'}`}
                    >
                      <span className={`absolute top-0.5 w-4 h-4 rounded-full bg-white transition-transform ${rowsFirst ? 'translate-x-4' : 'translate-x-0.5'}`} />
                    </button>
                    <span className={rowsFirst ? 'text-slate-300' : 'text-slate-500'}>rows first</span>
                  </div>
                  <button
                    onClick={handleExtract}
                    disabled={visibleRows.length === 0 || visibleCols.length === 0}
                    className="px-3 py-1 text-xs rounded bg-emerald-700 hover:bg-emerald-600 disabled:opacity-40 disabled:cursor-not-allowed text-white transition-colors"
                  >
                    {copied === 'extract' ? '✓ copied' : 'Extract →'}
                  </button>
                </div>

                {/* Bin table — checkboxes on headers */}
                <div className="flex-1 overflow-auto p-4">
                  <table className="text-xs border-collapse w-full table-fixed">
                    <colgroup>
                      <col className="w-36" />
                      {colVals.map((cv) => <col key={cv} />)}
                    </colgroup>
                    <thead>
                      <tr>
                        <th className="px-3 py-2 text-left text-slate-500 border border-slate-700 bg-slate-800 font-mono break-words">
                          {param1} ╲ {param2}
                        </th>
                        {colVals.map((cv) => (
                          <th key={cv}
                            className={`px-3 py-2 border border-slate-700 bg-slate-800 font-mono break-words transition-colors ${
                              selectedCols.has(cv) ? 'text-slate-300' : 'text-slate-600'
                            }`}
                          >
                            <label className="flex items-center gap-1.5 cursor-pointer justify-center">
                              <input
                                type="checkbox"
                                checked={selectedCols.has(cv)}
                                onChange={() => toggleCol(cv)}
                                className="accent-emerald-500 cursor-pointer"
                              />
                              <span>{cv}</span>
                            </label>
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {rowVals.map((rv) => (
                        <tr key={rv} className={selectedRows.has(rv) ? 'hover:bg-slate-800/30' : 'opacity-40'}>
                          <td className="px-3 py-2 border border-slate-700 bg-slate-800/50 font-mono break-words">
                            <label className="flex items-center gap-1.5 cursor-pointer">
                              <input
                                type="checkbox"
                                checked={selectedRows.has(rv)}
                                onChange={() => toggleRow(rv)}
                                className="accent-emerald-500 cursor-pointer shrink-0"
                              />
                              <span className={selectedRows.has(rv) ? 'text-slate-400' : 'text-slate-600'}>{rv}</span>
                            </label>
                          </td>
                          {colVals.map((cv) => (
                            <td key={cv} className="px-3 py-2 border border-slate-700 align-top">
                              {binTable[rv][cv].length > 0 ? (
                                <div className="space-y-0.5">
                                  {binTable[rv][cv].map((id) => (
                                    <button
                                      key={id}
                                      onClick={() => {
                                        const run = comparisonRuns.find((r) => r.id === id)
                                        if (run) setSelectedRun(resolvedSelected?.id === id ? null : run)
                                      }}
                                      className={`block font-mono text-xs text-left break-all w-full transition-colors ${
                                        resolvedSelected?.id === id ? 'text-emerald-400' : 'text-slate-300 hover:text-emerald-400'
                                      }`}
                                    >
                                      {id}
                                    </button>
                                  ))}
                                </div>
                              ) : (
                                <span className="text-slate-700">—</span>
                              )}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </>
            ) : (
              <div className="flex-1 flex items-center justify-center text-slate-600 text-sm">
                Select row and column parameters above
              </div>
            )}
          </>
        )}
      </div>

      {/* ── Right: config detail pane ── */}
      {resolvedSelected && (
        <ConfigDetail run={resolvedSelected} onClose={() => setSelectedRun(null)} />
      )}
    </div>
  )
}
