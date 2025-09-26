from typing import List, Dict, Tuple
from tensornetwork import Node, Edge
import math
import numpy as np


def _edge_dim(e: Edge) -> int:
    return int(e.dimension)

def _label_edges(nodes: List[Node]) -> Tuple[Dict[int, str], Dict[str, int]]:
    """
    Give each Edge a stable label based on id(edge).
    Labels are compact: e0, e1, ...
    """
    edge2label: Dict[int, str] = {}
    label2dim: Dict[str, int] = {}
    for n in nodes:
        for e in n.edges:
            k = id(e)
            if k not in edge2label:
                lab = f"e{len(edge2label)}"
                edge2label[k] = lab
                label2dim[lab] = _edge_dim(e)
    return edge2label, label2dim

def _node_sig(n: Node, edge2label: Dict[int, str]) -> Tuple[str, Tuple[str, ...]]:
    """Return (name, indices) for a node."""
    nm = getattr(n, "name", f"N{id(n)%1000}")
    idxs = tuple(edge2label[id(e)] for e in n.edges)
    return nm, idxs

def _prod(vals):
    p = 1
    for v in vals:
        p *= int(v)
    return p

def _pair_cost(indices_a: Tuple[str, ...], indices_b: Tuple[str, ...], dim: Dict[str, int]) -> Tuple[int, Tuple[str, ...]]:
    """
    Cost of contracting two tensors that share at least one index.
    Returns (cost, resulting_indices). If not connected, cost=inf.
    """
    A, B = set(indices_a), set(indices_b)
    shared = A & B
    if not shared:
        return math.inf, tuple()
    cost = _prod(dim[i] for i in indices_a) * _prod(dim[j] for j in indices_b) // _prod(dim[s] for s in shared)
    out = tuple(sorted((A | B) - shared))
    return cost, out

def _enumerate_all(tensors: List[Tuple[str, Tuple[str, ...]]],
                   dim: Dict[str, int]) -> List[Tuple[str, int]]:
    """
    Recursively enumerate all valid pairwise contraction orders.
    Returns list of (path_string, total_cost).
    """
    n = len(tensors)
    if n == 1:
        name, _ = tensors[0]
        return [(name, 0)]

    results: List[Tuple[str, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            name_i, idxs_i = tensors[i]
            name_j, idxs_j = tensors[j]
            step_cost, out_idxs = _pair_cost(idxs_i, idxs_j, dim)
            if step_cost == math.inf:
                continue  # not connected, skip

            merged_name = f"({name_i}×{name_j})"
            merged_tensor = (merged_name, out_idxs)

            # next tensor list
            next_list = [tensors[k] for k in range(n) if k not in (i, j)]
            next_list.append(merged_tensor)

            # enumerate downstream orders
            for sub_path, sub_cost in _enumerate_all(next_list, dim):
                total = step_cost + sub_cost
                # stitch a simple left-assoc string
                path_str = f"{merged_name}->{sub_path}" if len(next_list) > 1 else merged_name
                results.append((path_str, total))
    return results

def enumerate_contractions(nodes: List[Node]) -> List[Tuple[str, int]]:
    """
    Main API: return ALL valid contraction orders as (path_str, total_cost).
    """
    if not nodes:
        return [("", 0)]
    edge2label, dim = _label_edges(nodes)
    tensors = [_node_sig(n, edge2label) for n in nodes]

    # quick check: at least one connected pair
    if len(tensors) > 1 and not any(set(tensors[i][1]) & set(tensors[j][1])
                                    for i in range(len(tensors))
                                    for j in range(i + 1, len(tensors))):
        return [("<disconnected network>", math.inf)]

    return _enumerate_all(tensors, dim)

def best_contraction(nodes: List[Node]) -> Tuple[str, int]:
    """
    Convenience: pick the minimal-cost path from enumerate_contractions.
    """
    all_paths = enumerate_contractions(nodes)
    return min(all_paths, key=lambda pc: pc[1])



if __name__ == "__main__":
    m = 4

    a = 50
    c = 70
    b = 20

    u1 = Node(np.random.normal(0,1,(a,m,b,m)), name="u1")
    u2 = Node(np.random.normal(0,1,(b,int(np.sqrt(c*m/b)),int(np.sqrt(b*m/c)),c,
                                    int(np.sqrt(b*m/c)), int(np.sqrt(c*m/b)))), name="u2")
    v1 = Node(np.random.normal(0,1,(m,int(np.sqrt(c*m/b)),m)), name="v1")
    v2 = Node(np.random.normal(0,1,(m,int(np.sqrt(b*m/c)),m)), name="v2")


    u1[1] ^ v1[0]
    u1[2] ^ u2[0]

    v1[2] ^ v2[0]
    v1[1] ^ u2[1]

    v2[1] ^ u2[2]





    # Suppose you created Nodes A, B, C with names set (Node(tensor, name="A"), etc.)
    result = enumerate_contractions([u1,u2,v1,v2])
    min = np.inf
    path_m = ""
    for path, res in result:
        if res < min:
            min = res
            path_m = path
        print(path, res)

    print("Min", min, path_m)