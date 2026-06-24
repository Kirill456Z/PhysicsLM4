import { useMutation } from '@tanstack/react-query'
import { api } from '../api/client'
import { useStore } from '../store'

export function useGenerateSample() {
  const { configYaml, selectedTask, selectedTabName, setCurrentSample } = useStore()
  return useMutation({
    mutationFn: async () => {
      if (selectedTabName) await api.saveTaskTab(selectedTabName, configYaml)
      return api.generateSample(configYaml, selectedTask)
    },
    onSuccess: (data) => setCurrentSample(data),
  })
}

export function useGenerateBatch(batchSize: number) {
  const { configYaml, selectedTask, selectedTabName, setBatch } = useStore()
  return useMutation({
    mutationFn: async () => {
      if (selectedTabName) await api.saveTaskTab(selectedTabName, configYaml)
      return api.generateBatch(configYaml, selectedTask, batchSize)
    },
    onSuccess: (data) => setBatch(data.samples),
  })
}

export function useValidateConfig() {
  const { configYaml } = useStore()
  return useMutation({
    mutationFn: () => api.validateConfig(configYaml),
  })
}
