import CodeMirror from '@uiw/react-codemirror'
import { yaml } from '@codemirror/lang-yaml'
import { oneDark } from '@codemirror/theme-one-dark'
import { useStore } from '../store'

export default function ConfigEditor() {
  const { configYaml, setConfigYaml } = useStore()

  return (
    <div className="h-full overflow-hidden">
      <CodeMirror
        value={configYaml}
        height="100%"
        extensions={[yaml()]}
        theme={oneDark}
        onChange={(value) => setConfigYaml(value)}
        basicSetup={{
          lineNumbers: true,
          foldGutter: true,
          highlightActiveLine: true,
          autocompletion: false,
        }}
      />
    </div>
  )
}
