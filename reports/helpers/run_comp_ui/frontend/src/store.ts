import { create } from 'zustand'
import type { RunInfo } from './types'

interface AppStore {
  filterYaml: string
  minorVersion: number
  searchResultRuns: RunInfo[]
  comparisonRuns: RunInfo[]

  setFilterYaml: (yaml: string) => void
  setMinorVersion: (v: number) => void
  setSearchResultRuns: (runs: RunInfo[]) => void
  addToComparison: (run: RunInfo) => void
  removeFromComparison: (id: string) => void
}

export const useStore = create<AppStore>((set) => ({
  filterYaml: '',
  minorVersion: -1,
  searchResultRuns: [],
  comparisonRuns: [],

  setFilterYaml: (yaml) => set({ filterYaml: yaml }),
  setMinorVersion: (v) => set({ minorVersion: v }),
  setSearchResultRuns: (runs) => set({ searchResultRuns: runs }),
  addToComparison: (run) =>
    set((s) =>
      s.comparisonRuns.some((r) => r.id === run.id)
        ? s
        : { comparisonRuns: [...s.comparisonRuns, run] },
    ),
  removeFromComparison: (id) =>
    set((s) => ({ comparisonRuns: s.comparisonRuns.filter((r) => r.id !== id) })),
}))
