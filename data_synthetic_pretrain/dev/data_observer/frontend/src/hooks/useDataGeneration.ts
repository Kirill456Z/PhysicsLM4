import { useMutation } from '@tanstack/react-query'
import { api } from '../api/client'
import { useStore } from '../store'

export function useGenerateSample() {
  const { configYaml, selectedTask, setCurrentSample } = useStore()
  return useMutation({
    mutationFn: () => api.generateSample(configYaml, selectedTask),
    onSuccess: (data) => setCurrentSample(data),
  })
}

export function useGenerateBatch(batchSize: number) {
  const { configYaml, selectedTask, setBatch } = useStore()
  return useMutation({
    mutationFn: () => api.generateBatch(configYaml, selectedTask, batchSize),
    onSuccess: (data) => setBatch(data.samples),
  })
}

export function useValidateConfig() {
  const { configYaml } = useStore()
  return useMutation({
    mutationFn: () => api.validateConfig(configYaml),
  })
}
