from typing import Set, Tuple
from networkx import MultiDiGraph
from scipy.sparse import lil_matrix
from pyformlang.finite_automaton import State
from collections import deque

from project.build_graph import regex_to_dfa, graph_to_nfa


def ms_bfs_based_rpq(
    regex: str, graph: MultiDiGraph, start_nodes: Set[int], final_nodes: Set[int]
) -> Set[Tuple[int, int]]:
    dfa = regex_to_dfa(regex)
    graph_nfa = graph_to_nfa(graph, start_nodes, final_nodes)

    graph_states = list(graph_nfa.states)
    dfa_states = list(dfa.states)

    n_graph = len(graph_states)
    n_dfa = len(dfa_states)

    graph_state_to_idx = {state: i for i, state in enumerate(graph_states)}
    dfa_state_to_idx = {state: i for i, state in enumerate(dfa_states)}

    graph_transitions = graph_nfa.to_dict()
    graph_matrices = {}
    for symbol in graph_nfa.symbols:
        matrix = lil_matrix((n_graph, n_graph), dtype=bool)
        for source_state in graph_states:
            if source_state not in graph_transitions:
                continue
            state_transitions = graph_transitions[source_state]
            if symbol not in state_transitions:
                continue
            dest_states = state_transitions[symbol]
            if not isinstance(dest_states, set):
                dest_states = {dest_states}
            src_idx = graph_state_to_idx[source_state]
            for dest_state in dest_states:
                dest_idx = graph_state_to_idx[dest_state]
                matrix[src_idx, dest_idx] = True
        graph_matrices[symbol] = matrix.tocsr()

    dfa_transitions = dfa.to_dict()
    dfa_matrices = {}
    for symbol in dfa.symbols:
        matrix = lil_matrix((n_dfa, n_dfa), dtype=bool)
        for src_state in dfa_states:
            if src_state not in dfa_transitions:
                continue
            if symbol not in dfa_transitions[src_state]:
                continue
            dest_state = dfa_transitions[src_state][symbol]
            src_idx = dfa_state_to_idx[src_state]
            dest_idx = dfa_state_to_idx[dest_state]
            matrix[src_idx, dest_idx] = True
        dfa_matrices[symbol] = matrix.tocsr()

    start_nodes_list = sorted(start_nodes)
    start_node_to_index = {node: idx for idx, node in enumerate(start_nodes_list)}

    R = lil_matrix((n_graph, n_dfa), dtype=int)
    queue = deque()

    for start_node in start_nodes_list:
        graph_idx = graph_state_to_idx[State(start_node)]
        for dfa_start in dfa.start_states:
            dfa_idx = dfa_state_to_idx[dfa_start]
            bit_mask = 1 << start_node_to_index[start_node]
            if (R[graph_idx, dfa_idx] & bit_mask) == 0:
                R[graph_idx, dfa_idx] |= bit_mask
                queue.append((graph_idx, dfa_idx))

    while queue:
        current_graph_idx, current_dfa_idx = queue.popleft()
        current_mask = R[current_graph_idx, current_dfa_idx]
        for symbol in graph_matrices.keys():
            if symbol not in dfa_matrices:
                continue
            next_graph_indices = (
                graph_matrices[symbol].getrow(current_graph_idx).indices
            )
            next_dfa_indices = dfa_matrices[symbol].getrow(current_dfa_idx).indices
            for next_graph_idx in next_graph_indices:
                for next_dfa_idx in next_dfa_indices:
                    existing_mask = R[next_graph_idx, next_dfa_idx]
                    new_mask = existing_mask | current_mask
                    if new_mask != existing_mask:
                        R[next_graph_idx, next_dfa_idx] = new_mask
                        queue.append((next_graph_idx, next_dfa_idx))

    result = set()
    for final_node in final_nodes:
        final_graph_idx = graph_state_to_idx[State(final_node)]
        for final_dfa_state in dfa.final_states:
            final_dfa_idx = dfa_state_to_idx[final_dfa_state]
            mask = R[final_graph_idx, final_dfa_idx]
            for i in range(len(start_nodes_list)):
                if mask & (1 << i):
                    result.add((start_nodes_list[i], final_node))
    return result
