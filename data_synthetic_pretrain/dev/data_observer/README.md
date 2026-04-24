# Synthetic Data Observer UI

## Overview

The Synthetic Data Observer UI is a web-based tool for examining and debugging synthetic task data generation from the `data_synthetic_pretrain` package. It provides interactive visualization of generated graphs, task-specific metadata, and the resulting formatted data (context, loss masks, labels) used for training.

### Key Capabilities

- **Graph Visualization**: Display generated graphs (DAG, directed, undirected) with interactive node/edge exploration
- **Task Inspection**: View generated tasks including:
  - Raw graph structure with node tokens and edges
  - Context (tokenized sequence), loss mask, and labels
  - Task-specific metadata (query nodes, answers, num_hops for depo)
- **Config Management**: Edit YAML configs in real-time and regenerate samples to observe impact
- **Multi-Task Support**: Separate tabs for each synthetic task type (depo, brevo, concomp, concomp_factor, shortest_path)
- **Batch Preview**: Generate and inspect batches with multiple samples

---

## Architecture

### High-Level Design

```
┌─────────────────────────────────────────────────────┐
│          React Frontend (Port 3000)                 │
│  ┌───────────────────────────────────────────────┐  │
│  │ Config Editor (Monaco/CodeMirror)             │  │
│  │ Task Tabs (Depo | Brevo | ConComp | ...)      │  │
│  │ Graph Visualization (Cytoscape.js)            │  │
│  │ Task Data Inspector (Context/Loss/Labels)     │  │
│  └───────────────────────────────────────────────┘  │
└────────────────────────┬──────────────────────────┘
                         │ REST API
                         ▼
┌─────────────────────────────────────────────────────┐
│       FastAPI Backend (Port 8000)                   │
│  ┌───────────────────────────────────────────────┐  │
│  │ Config Parser (custom, YAML only)             │  │
│  │ Task Generation Endpoints                     │  │
│  │ Batch Generation & Formatting                 │  │
│  │ Data Export (JSON/CSV)                        │  │
│  └───────────────────────────────────────────────┘  │
│  🚫 NO imports from lingua_modified              │
│  ✅ ONLY imports from data_synthetic_pretrain    │
└────────────────────────┬──────────────────────────┘
                         │ (lightweight boundary)
                         ▼
         ┌──────────────────────────────────┐
         │  data_synthetic_pretrain Package  │
         │  - Task Generators               │
         │  - Dataloader & Formatting       │
         │  - Graph Generation              │
         │  ✅ No torch/cuda/xformers here   │
         └──────────────────────────────────┘
```

**Key Design Decision**: The observer runs in a lightweight, isolated environment separate from the training infrastructure. This enables deployment on any machine (laptop, CI/CD, cloud) without GPU dependencies.

### Component Structure

**Frontend**:
- `src/App.tsx`: Main app shell with tabs
- `src/pages/TaskViewer.tsx`: Per-task view
- `src/components/GraphViewer.tsx`: Cytoscape graph visualization
- `src/components/ConfigEditor.tsx`: YAML config editor with validation
- `src/components/DataInspector.tsx`: Context/loss_mask/labels display
- `src/components/TaskMetadata.tsx`: Task-specific info (query_nodes, answer_nodes, etc.)
- `src/hooks/useDataGeneration.ts`: React Query hook for API calls
- `src/api/client.ts`: API client functions

**Backend**:
- `backend/main.py`: FastAPI app entry point
- `backend/api/routes.py`: API endpoints
- `backend/models.py`: Pydantic request/response models
- `backend/generation.py`: Task generation logic wrapper
- `backend/config.py`: Config loading and validation

---

## Framework & Library Selection

### Frontend Framework: **React 18** with **Vite**

**Rationale**:
- Fast development with hot module replacement
- Strong ecosystem for visualization and state management
- TypeScript support for type safety
- Mature and well-documented

**Key Dependencies**:
- **react-query** (TanStack Query): Async state management for API calls
- **zustand**: Lightweight state management for UI state (config, selected task)
- **cytoscape.js** + **react-cytoscapejs**: Graph visualization
  - Perfect for DAGs and directed graphs
  - Flexible layout algorithms (hierarchical for DAGs)
  - Performance: handles 100+ nodes smoothly
  - Interactive: pan, zoom, click-to-select nodes/edges
- **monaco-editor** or **@uiw/react-codemirror**: YAML config editor
  - Monaco: VSCode-like experience, heavy (~5MB)
  - CodeMirror: Lighter (~200KB), sufficient for config editing
  - **Recommendation**: CodeMirror for quick iteration
- **react-tabs** or **@headlessui/react**: Tab navigation
- **tailwindcss**: Utility-first CSS for rapid styling
- **vite-plugin-raw**: Import YAML files as text
- **@monaco-editor/loader** (if using Monaco): Async loader for editor

### Backend Framework: **FastAPI**

**Rationale**:
- Lightweight and focused (unlike Django)
- Automatic OpenAPI/Swagger docs
- Built-in async support
- Excellent validation with Pydantic (which the package already uses)

**Key Dependencies** (intentionally minimal, no torch/cuda):
- **fastapi**: Web framework
- **uvicorn**: ASGI server
- **pydantic**: Config validation
- **pyyaml**: YAML parsing
- **python-cors**: CORS support for dev
- **numpy**: Data handling (already in data_synthetic_pretrain)

**Note**: Backend imports ONLY from `data_synthetic_pretrain`, NOT from `lingua_modified`. This keeps the observer lightweight and self-contained without requiring torch, cuda, or xformers.

### Graph Visualization: **Cytoscape.js**

**Comparison**:
| Library | Pros | Cons | Best For |
|---------|------|------|----------|
| **Cytoscape.js** | DAG layouts, performant, rich styling | Steeper learning curve | DAGs, directed graphs ✅ |
| Vis.js | Physics engine, intuitive | Heavy bundle | General networks |
| Force-Graph | 3D support, beautiful | Overkill for this use case | Large complex networks |
| D3.js | Maximum flexibility | Very verbose | Custom requirements |

**Recommendation**: Cytoscape.js for its excellent DAG support and performance.

**React Wrapper**: Use `react-cytoscapejs` to integrate Cytoscape into React components.

### Config Editor: **CodeMirror 6**

**Rationale**:
- Lightweight (~200KB)
- Excellent YAML support via language mode
- Syntax highlighting, line numbers, bracket matching
- `@uiw/react-codemirror` provides React integration
- Alternative: Monaco if full VSCode experience is desired

---

## API Design

### Base URL
`http://localhost:8000/api`

### Endpoints

#### 1. **GET** `/tasks` - List available task types
```json
Response: {
  "tasks": ["depo", "brevo", "concomp", "concomp_factor", "shortest_path"]
}
```

#### 2. **POST** `/validate-config` - Validate YAML config
```json
Request: {
  "config_yaml": "name: test\nsteps: 100\n..."
}

Response (Success): {
  "valid": true,
  "parsed": { /* parsed config object */ }
}

Response (Error): {
  "valid": false,
  "errors": ["Field 'steps' is required", ...]
}
```

#### 3. **POST** `/generate-sample` - Generate single task sample
```json
Request: {
  "config_yaml": "...",
  "task_name": "depo"
}

Response: {
  "task": {
    "graph": {
      "nodes": [{"tokens": [1, 2, 3]}, ...],
      "edges": [{"from": 0, "to": 1}, ...],
      "n_nodes": 5
    },
    "task_index": 100,
    "context": [100, 1, 2, ...],
    "loss_mask": [0, 0, 1, ...],
    "labels": [1, 2, ..., -100],  // After formatting
    "depo_specific": {              // Task-specific fields
      "query_nodes": [[1,2,3], ...],
      "answer_nodes": [[4,5], ...],
      "num_hops": [1, 2, ...]
    }
  }
}
```

#### 4. **POST** `/generate-batch` - Generate batch of samples
```json
Request: {
  "config_yaml": "...",
  "task_name": "depo",
  "batch_size": 5
}

Response: {
  "samples": [/* array of task objects */],
  "batch_shape": [5, 1024, 2]  // (batch_size, seq_len, 2)
}
```

#### 5. **POST** `/export-sample` - Export sample as JSON/CSV
```json
Request: {
  "config_yaml": "...",
  "task_name": "depo",
  "format": "json"  // or "csv"
}

Response: file download
```

---

## Implementation Plan

### Phase 1: Backend API (Weeks 1-2)

1. **Setup FastAPI server**
   - Create `backend/main.py` with app factory
   - Configure CORS for dev (localhost:3000)
   - Add uvicorn run configuration

2. **Implement lightweight config parser**
   - Create `backend/config_parser.py` with custom Pydantic models
   - Extract only fields needed for data generation (synthetic_tasks_generation_args, data.seq_len, data.batch_size, etc.)
   - **Do NOT import from lingua_modified** — parse YAML directly
   - Validate YAML structure and types
   - Return parsed config or validation errors
   - See [Config Handling](#config-handling) section below for implementation details

3. **Implement task generation endpoints**
   - Load task generator from `data_synthetic_pretrain` (ONLY this package)
   - Generate single sample with graph/context/loss_mask
   - Format using SyntheticDataLoader logic
   - Extract task-specific fields (DepoSyntheticTask, etc.)
   - Return JSON response

4. **Add data serialization**
   - Create Pydantic models for API responses
   - Handle Graph → JSON serialization (nodes as token tuples)
   - Handle numpy arrays → lists
   - Ensure all output is JSON-serializable

### Phase 2: Frontend - Core UI (Weeks 2-3)

1. **Setup React + Vite project**
   - Create `frontend/` directory
   - Initialize Vite + React + TypeScript
   - Setup TailwindCSS + ESlint

2. **Build main application shell**
   - Create `App.tsx` with task type tabs
   - Setup React Query client
   - Initialize zustand store for UI state (selected task, config)

3. **Implement Config Editor**
   - CodeMirror component with YAML syntax highlighting
   - Validation button → calls `/validate-config` endpoint
   - Show validation errors inline
   - Preload with `depo_debug.yaml` on startup
   - Save/load configs from localStorage

4. **Implement Task Data Inspector**
   - Component to display:
     - Task metadata (task_index, num_nodes, num_edges)
     - Context tokens (with token ID and interpretation)
     - Loss mask visualization (0 = no loss, 1 = compute loss)
     - Labels (with special token handling)
   - Tabular view or side-by-side tokenization
   - Token search/filter

### Phase 3: Frontend - Graph Visualization (Week 3)

1. **Implement Graph Viewer**
   - Cytoscape component to render graph:
     - Nodes: labeled with token IDs, size by degree
     - Edges: directed, color-coded if needed
     - Layout: hierarchical for DAGs, force-directed fallback
   - Interactive features:
     - Click node → highlight in token viewer
     - Zoom/pan controls
     - Legend (token colors, edge meanings)

2. **Link graph to token display**
   - Clicking node in graph highlights its tokens in context
   - Clicking token in context highlights node in graph

### Phase 4: Frontend - Task-Specific Views (Week 4)

1. **Task Metadata Component**
   - Depo: Show query_nodes, answer_nodes, num_hops with highlighting
   - Brevo/ConComp: Show task-specific fields
   - Display in table or structured format

2. **Batch Preview**
   - Generate multiple samples
   - Show sample selector (1 of 5)
   - Comparison view for parameter sensitivity

### Phase 5: Polish & Testing (Week 5)

1. **Error handling & loading states**
   - Loading spinners during API calls
   - Error messages with retry buttons
   - Validation feedback

2. **Responsiveness**
   - Mobile-friendly layout (though primarily desktop tool)
   - Resizable panes

3. **Integration testing**
   - End-to-end flow: edit config → validate → generate → visualize

---

## Directory Structure

```
data_synthetic_pretrain/dev/data_observer/
├── README.md                    (this file)
├── backend/
│   ├── main.py                 (FastAPI app entry)
│   ├── requirements.txt         (backend deps: fastapi, uvicorn, pyyaml, etc.)
│   ├── api/
│   │   └── routes.py           (API endpoints)
│   ├── models.py               (Pydantic request/response schemas)
│   ├── generation.py           (Wrapper around data_synthetic_pretrain)
│   └── config.py               (Config loading, path resolution)
├── frontend/
│   ├── package.json            (React deps, scripts)
│   ├── vite.config.ts
│   ├── tsconfig.json
│   ├── tailwind.config.js
│   ├── src/
│   │   ├── App.tsx
│   │   ├── index.css           (TailwindCSS imports)
│   │   ├── pages/
│   │   │   └── TaskViewer.tsx  (Per-task view)
│   │   ├── components/
│   │   │   ├── GraphViewer.tsx        (Cytoscape)
│   │   │   ├── ConfigEditor.tsx       (CodeMirror)
│   │   │   ├── DataInspector.tsx      (Context/Loss/Labels)
│   │   │   └── TaskMetadata.tsx       (Task-specific fields)
│   │   ├── hooks/
│   │   │   └── useDataGeneration.ts   (React Query hooks)
│   │   ├── api/
│   │   │   └── client.ts             (Fetch wrapper)
│   │   ├── store.ts                  (Zustand store)
│   │   └── types.ts                  (TypeScript types)
│   └── public/
└── docker-compose.yml          (optional: for containerized dev)
```

---

## Development Setup

### Prerequisites
- Python 3.12+
- Node 18+
- Poetry (Python) or pip

### Backend Setup

```bash
cd data_synthetic_pretrain/dev/data_observer/backend

# Create virtual env
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install deps (lightweight, no torch/cuda/xformers)
pip install -r requirements.txt

# Minimal requirements.txt contents:
# fastapi==0.104.1
# uvicorn[standard]==0.24.0
# pydantic==2.5.0
# pyyaml==6.0
# python-cors==4.0.0
# numpy>=1.24

# Run server
uvicorn main:app --reload --port 8000
```

**Note**: This creates an isolated, lightweight environment independent of the main project. No torch, cuda, or xformers needed.

### Frontend Setup

```bash
cd data_synthetic_pretrain/dev/data_observer/frontend

# Install deps
npm install

# Dev server (hot reload on port 3000)
npm run dev

# Build for production
npm run build
```

### Accessing the UI
- **Frontend**: http://localhost:3000
- **API Docs**: http://localhost:8000/docs (Swagger)

---

## Key Implementation Details

### Graph Serialization

Graphs must be JSON-serializable for API responses. NodeWord uses frozen Pydantic models, which serialize to JSON automatically. Edges dict must be converted to list format:

```python
# From: dict[NodeWord, List[NodeWord]]
# To: [{"from": node_idx, "to": neighbor_idx, ...}]
def serialize_graph(graph: Graph):
    node_index = {node: i for i, node in enumerate(graph.nodes)}
    edges = [
        {"from": node_index[src], "to": node_index[dst]}
        for src, neighbors in graph.edges.items()
        for dst in neighbors
    ]
    return {
        "nodes": [{"tokens": list(n.tokens)} for n in graph.nodes],
        "edges": edges,
        "n_nodes": graph.n
    }
```

### Config Handling (Lightweight, No lingua_modified)

Create a custom config parser that extracts only what's needed for data generation:

**File**: `backend/config_parser.py`

```python
from pydantic import BaseModel, field_validator
from typing import Optional, Any
import yaml

class SyntheticTaskConfig(BaseModel):
    task_name: str
    generation_args: dict[str, Any]
    weight: float = 1.0

class SyntheticTasksGenerationArgs(BaseModel):
    synthetic_tasks: list[SyntheticTaskConfig] = []

class DataObserverConfig(BaseModel):
    """Lightweight config for data observer (only fields needed for generation)"""
    
    synthetic_tasks_generation_args: SyntheticTasksGenerationArgs
    seq_len: int = 1024
    batch_size: int = 32
    pad_token: int = 0
    no_train_label_token: int = -100
    n_workers: int = 1  # Single worker for observer
    
    @field_validator('synthetic_tasks_generation_args', mode='before')
    @classmethod
    def parse_synthetic_tasks(cls, v):
        if isinstance(v, dict):
            return SyntheticTasksGenerationArgs(**v)
        return v

@staticmethod
def load_config(config_yaml: str) -> DataObserverConfig:
    """Parse YAML config without importing lingua_modified"""
    config_dict = yaml.safe_load(config_yaml)
    
    # Extract only fields needed for data generation
    extracted = {
        'synthetic_tasks_generation_args': config_dict.get('synthetic_tasks_generation_args', {}),
        'seq_len': config_dict.get('data', {}).get('seq_len', 1024),
        'batch_size': config_dict.get('data', {}).get('batch_size', 32),
        'pad_token': config_dict.get('data', {}).get('pad_token', 0),
        'no_train_label_token': config_dict.get('data', {}).get('no_train_label_token', -100),
    }
    
    return DataObserverConfig(**extracted)
```

**Usage in API**:

```python
from backend.config_parser import load_config
from data_synthetic_pretrain.tasks import SYNTHETIC_TASKS
from data_synthetic_pretrain.dataloader.data_generation_args import (
    SyntheticTasksFormattingArgs
)

# In endpoint:
config = load_config(config_yaml)

# Extract specific task generator
task_config = next(
    t for t in config.synthetic_tasks_generation_args.synthetic_tasks
    if t.task_name == "depo"
)

# Instantiate generator (from data_synthetic_pretrain only)
generator = SYNTHETIC_TASKS["depo"].build_from_dict(task_config.generation_args)

# Format output
formatting_args = SyntheticTasksFormattingArgs(
    batch_size=config.batch_size,
    seq_len=config.seq_len,
    pad_token=config.pad_token,
    no_train_label_token=config.no_train_label_token,
    n_workers=1,
)
```

**Key Benefits**:
- ✅ No torch/cuda/xformers imports
- ✅ Lightweight validation with Pydantic
- ✅ Only imports from `data_synthetic_pretrain`
- ✅ Can run in isolated environment without main project dependencies

### Frontend State Management

Use Zustand for lightweight state:

```typescript
import create from 'zustand';

interface Store {
  selectedTask: string;
  currentConfig: string;
  setSelectedTask: (task: string) => void;
  setCurrentConfig: (config: string) => void;
}

export const useStore = create<Store>(set => ({
  selectedTask: 'depo',
  currentConfig: '',
  setSelectedTask: (task) => set({ selectedTask: task }),
  setCurrentConfig: (config) => set({ currentConfig: config })
}));
```

---

## Future Enhancements

1. **Batch Comparison**: Compare multiple generated samples side-by-side
2. **Parameter Sweep**: Automated generation with parameter variations (e.g., `max_hops: [1, 5, 10]`) to observe impact
3. **Performance Profiling**: Measure generation time, dataloader throughput
4. **Export to Dataset**: Save generated batches as HDF5/PyArrow for training
5. **Task Composition**: Generate mixed tasks (multi-task batches) with visualization
6. **Tokenizer Integration**: Display token interpretations (BOS, EOS, task tokens)
7. **Real-time Dataloader**: Stream from actual dataloader during training (debug mode)
8. **Dark Mode**: Theme toggle for accessibility

---

## Testing Strategy

### Backend Tests
- Unit tests for config validation
- Integration tests for task generation with various configs
- API endpoint tests (mocked dataloader)

### Frontend Tests
- Component tests (React Testing Library)
- API client mocking (MSW)
- E2E tests for critical flows (edit config → generate → visualize)

---

## Performance Considerations

- **Graph Rendering**: Cytoscape handles 100+ node graphs smoothly; cache large graphs
- **Task Generation**: Limit batch size to 10 samples per request (adjust API POST based on testing)
- **API Calls**: Debounce config changes (e.g., 500ms) before re-generating to avoid spam
- **Frontend Bundle**: Tree-shake unused Cytoscape features; lazy-load editor

---

---

## Dependency Isolation Strategy

The observer is designed to run independently without pulling in heavy training dependencies:

### ✅ Imports ALLOWED in Backend
- `data_synthetic_pretrain.*` — task generation, graphs, dataloader
- `pydantic`, `fastapi`, `uvicorn` — web framework
- `numpy`, `pyyaml` — data handling
- Standard library — `json`, `io`, `contextlib`, etc.

### ❌ Imports FORBIDDEN
- `lingua_modified.*` — brings torch, cuda, xformers
- `torch`, `transformers` — training dependencies
- `xformers`, `flash_attn` — CUDA extensions

### Why This Matters
- Observers can run on laptops or CI environments without GPU
- Faster startup (no torch compilation)
- Smaller Docker images if containerized
- Clear separation of concerns

### If You Need Training Config
Instead of importing `TrainArgs`:
```python
# ❌ DON'T DO THIS
from lingua_modified.apps.main.train_args import TrainArgs

# ✅ DO THIS
# Parse YAML directly and extract fields needed for data generation
# See config_parser.py example above
```

---

## References

- **Cytoscape.js Docs**: https://js.cytoscape.org/
- **FastAPI Tutorial**: https://fastapi.tiangolo.com/
- **React Query Docs**: https://tanstack.com/query/latest
- **CodeMirror Docs**: https://codemirror.net/
- **Pydantic Docs**: https://docs.pydantic.dev/
