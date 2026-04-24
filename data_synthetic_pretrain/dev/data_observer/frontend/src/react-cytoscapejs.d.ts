declare module 'react-cytoscapejs' {
  import * as React from 'react'
  import type * as cytoscape from 'cytoscape'

  interface CytoscapeComponentProps {
    elements: cytoscape.ElementDefinition[]
    stylesheet?: cytoscape.StylesheetStyle[]
    layout?: cytoscape.LayoutOptions
    style?: React.CSSProperties
    cy?: (cy: cytoscape.Core) => void
    key?: React.Key
    className?: string
    id?: string
  }

  const CytoscapeComponent: React.FC<CytoscapeComponentProps>
  export default CytoscapeComponent
}
