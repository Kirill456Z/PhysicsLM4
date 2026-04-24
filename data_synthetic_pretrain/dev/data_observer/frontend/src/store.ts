import { create } from 'zustand'
import type { SampleResponse } from './types'

type ResultTab = 'graph' | 'tokens' | 'metadata' | 'eval'

interface AppStore {
  selectedTask: string
  configYaml: string
  currentSample: SampleResponse | null
  batchSamples: SampleResponse[]
  batchIndex: number
  activeResultTab: ResultTab

  setSelectedTask: (task: string) => void
  setConfigYaml: (yaml: string) => void
  setCurrentSample: (sample: SampleResponse | null) => void
  setBatch: (samples: SampleResponse[]) => void
  setBatchIndex: (i: number) => void
  setActiveResultTab: (tab: ResultTab) => void
}

export const useStore = create<AppStore>((set) => ({
  selectedTask: 'depo',
  configYaml: '',
  currentSample: null,
  batchSamples: [],
  batchIndex: 0,
  activeResultTab: 'graph',

  setSelectedTask: (task) => set({ selectedTask: task, currentSample: null, batchSamples: [], batchIndex: 0 }),
  setConfigYaml: (yaml) => set({ configYaml: yaml }),
  setCurrentSample: (sample) => set({ currentSample: sample }),
  setBatch: (samples) => set({ batchSamples: samples, batchIndex: 0, currentSample: samples[0] ?? null }),
  setBatchIndex: (i) => set((s) => ({ batchIndex: i, currentSample: s.batchSamples[i] ?? null })),
  setActiveResultTab: (tab) => set({ activeResultTab: tab }),
}))
