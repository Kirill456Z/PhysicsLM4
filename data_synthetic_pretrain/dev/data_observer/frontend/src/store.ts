import { create } from 'zustand'
import type { SampleResponse, TaskTab } from './types'

type ResultTab = 'graph' | 'tokens' | 'metadata'

interface AppStore {
  tabs: TaskTab[]
  selectedTabName: string
  selectedTask: string
  configYaml: string
  currentSample: SampleResponse | null
  batchSamples: SampleResponse[]
  batchIndex: number
  activeResultTab: ResultTab

  setTabs: (tabs: TaskTab[]) => void
  setActiveTab: (tab: TaskTab) => void
  setConfigYaml: (yaml: string) => void
  setCurrentSample: (sample: SampleResponse | null) => void
  setBatch: (samples: SampleResponse[]) => void
  setBatchIndex: (i: number) => void
  setActiveResultTab: (tab: ResultTab) => void
}

export const useStore = create<AppStore>((set) => ({
  tabs: [],
  selectedTabName: '',
  selectedTask: 'depo',
  configYaml: '',
  currentSample: null,
  batchSamples: [],
  batchIndex: 0,
  activeResultTab: 'graph',

  setTabs: (tabs) =>
    set((s) => {
      // Keep existing selection if possible; otherwise select first tab.
      const current = tabs.find((t) => t.tab_name === s.selectedTabName) ?? tabs[0]
      if (!current) return { tabs }
      return {
        tabs,
        selectedTabName: current.tab_name,
        selectedTask: current.task_name,
        configYaml: current.config_yaml,
        currentSample: null,
        batchSamples: [],
        batchIndex: 0,
      }
    }),
  setActiveTab: (tab) =>
    set({
      selectedTabName: tab.tab_name,
      selectedTask: tab.task_name,
      configYaml: tab.config_yaml,
      currentSample: null,
      batchSamples: [],
      batchIndex: 0,
    }),
  setConfigYaml: (yaml) => set({ configYaml: yaml }),
  setCurrentSample: (sample) => set({ currentSample: sample }),
  setBatch: (samples) => set({ batchSamples: samples, batchIndex: 0, currentSample: samples[0] ?? null }),
  setBatchIndex: (i) => set((s) => ({ batchIndex: i, currentSample: s.batchSamples[i] ?? null })),
  setActiveResultTab: (tab) => set({ activeResultTab: tab }),
}))
