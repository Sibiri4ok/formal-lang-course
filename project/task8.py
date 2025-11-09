import networkx as nx
from pyformlang.cfg import CFG
from pyformlang.rsa import RecursiveAutomaton
from pyformlang.finite_automaton import State, NondeterministicFiniteAutomaton
from scipy.sparse import dok_matrix

from project.build_graph import graph_to_nfa
from project.tensor_automata import AdjacencyMatrixFA, intersect_automata


def rsm_to_nfa(rsm: RecursiveAutomaton) -> NondeterministicFiniteAutomaton:
    nfa = NondeterministicFiniteAutomaton()

    for var, box in rsm.boxes.items():
        dfa = box.dfa
        starts_and_finals = dfa.start_states | dfa.final_states

        for state in starts_and_finals:
            new_state = State((var, state))
            if state in dfa.start_states:
                nfa.add_start_state(new_state)
            if state in dfa.final_states:
                nfa.add_final_state(new_state)

        graph = dfa.to_networkx()
        for src, dest, label in graph.edges(data="label"):
            nfa.add_transition(State((var, src)), label, State((var, dest)))
    return nfa


def tensor_based_cfpq(
    rsm: RecursiveAutomaton,
    graph: nx.DiGraph,
    start_nodes: set[int] = None,
    final_nodes: set[int] = None,
) -> set[tuple[int, int]]:
    graph_m = AdjacencyMatrixFA(
        graph_to_nfa(nx.MultiDiGraph(graph), start_nodes, final_nodes)
    )
    graph_idx_to_state = graph_m.index_to_state
    rsm_m = AdjacencyMatrixFA(rsm_to_nfa(rsm))

    for non_term in rsm.boxes:
        for automaton in (graph_m, rsm_m):
            automaton.transition_matrices.setdefault(
                non_term,
                dok_matrix((len(automaton.states), len(automaton.states)), dtype=bool),
            )

    prev_nnz, curr_nnz = -1, 0
    while prev_nnz != curr_nnz:
        prev_nnz = curr_nnz
        intersection = intersect_automata(rsm_m, graph_m)
        idx_to_state = {
            idx: state for state, idx in intersection.state_to_index.items()
        }

        srcs, dests = intersection.get_trans_closure().nonzero()
        for s_idx, d_idx in zip(srcs, dests):
            src_rsm_state, src_graph_node = idx_to_state[s_idx].value
            src_symbol, src_rsm_node = src_rsm_state.value
            dest_rsm_state, dest_graph_node = idx_to_state[d_idx].value
            dest_symbol, dest_rsm_node = dest_rsm_state.value

            if src_symbol != dest_symbol:
                continue

            src_rsm_states = rsm.boxes[src_symbol].dfa.start_states
            dest_rsm_states = rsm.boxes[src_symbol].dfa.final_states

            if src_rsm_node in src_rsm_states and dest_rsm_node in dest_rsm_states:
                graph_m.transition_matrices[src_symbol][
                    graph_m.state_to_index[src_graph_node],
                    graph_m.state_to_index[dest_graph_node],
                ] = True

        curr_nnz = sum(
            matrix.count_nonzero() for _, matrix in graph_m.transition_matrices.items()
        )

    result = {
        (graph_idx_to_state[start].value, graph_idx_to_state[final].value)
        for start in graph_m.start_states
        for final in graph_m.final_states
        if graph_m.transition_matrices[rsm.initial_label][start, final]
    }
    return result


def cfg_to_rsm(cfg: CFG) -> RecursiveAutomaton:
    return ebnf_to_rsm(cfg.to_text())


def ebnf_to_rsm(ebnf: str) -> RecursiveAutomaton:
    return RecursiveAutomaton.from_text(ebnf)
