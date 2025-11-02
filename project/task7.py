from collections import defaultdict
from itertools import product
from typing import Set
from typing import Tuple

from scipy.sparse import csr_matrix
from project.task6 import cfg_to_weak_normal_form

import networkx as nx
import pyformlang


def matrix_based_cfpq(
    cfg: pyformlang.cfg.CFG,
    graph: nx.DiGraph,
    start_nodes: Set[int] = None,
    final_nodes: Set[int] = None,
) -> Set[Tuple[int, int]]:
    start_nodes = start_nodes or set(graph.nodes)
    final_nodes = final_nodes or set(graph.nodes)

    node_to_idx = {node: i for i, node in enumerate(graph.nodes)}
    n = graph.number_of_nodes()

    cfg = cfg_to_weak_normal_form(cfg)

    matrices = {nt.value: csr_matrix((n, n), dtype=bool) for nt in cfg.variables}

    term_to_vars = defaultdict(list)
    for rule in cfg.productions:
        if len(rule.body) > 0 and isinstance(rule.body[0], pyformlang.cfg.Terminal):
            term_to_vars[rule.body[0].value].append(rule.head.value)

    for u, v, data in graph.edges(data=True):
        label = data.get("label")
        if not label:
            continue
        ui, vi = node_to_idx[u], node_to_idx[v]
        for var in term_to_vars.get(label, []):
            matrices[var][ui, vi] = True

    for non_term in cfg.get_nullable_symbols():
        var = non_term.value
        for i in range(n):
            matrices[var][i, i] = True

    binary_rules = {
        tuple(map(lambda x: x.value, [rule.head, rule.body[0], rule.body[1]]))
        for rule in cfg.productions
        if len(rule.body) == 2
    }

    is_changed = True
    while is_changed:
        is_changed = False
        for A, B, C in binary_rules:
            before = matrices[A].nnz
            matrices[A] = (matrices[A] + matrices[B] @ matrices[C]).astype(bool)
            if matrices[A].nnz > before:
                is_changed = True

    start_symbol = cfg.start_symbol.value
    result = {
        (u, v)
        for u, v in product(start_nodes, final_nodes)
        if matrices[start_symbol][node_to_idx[u], node_to_idx[v]]
    }

    return result
