### Synthetic Pretrain Tasks

- `depo.py` (Depth): k-hop reasoning on directed circular permutations. Input provides graph edges and queries (`k`, start node); target is the k-th successor node(s).
- `brevo.py` (Breadth): reachable-subgraph reasoning in directed acyclic graphs. Given a query node, target is all reachable nodes in valid topological order.
- `concomp.py` (Connectivity): BFS-style traversal over directed random graphs. For each query node, target is the set/order of discovered reachable nodes.
- `concomp_factor.py` (Component Factorization): connected-component reasoning on undirected graphs. Target is one canonical representative per connected component.
- `shortest_path.py` (Path Planning): shortest-path reasoning on directed random graphs. Query gives start/end nodes; target is the node sequence of the BFS shortest path.