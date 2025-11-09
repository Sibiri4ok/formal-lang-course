from typing import Iterable
from itertools import product
from networkx import MultiDiGraph
from pyformlang.finite_automaton import State, Symbol, NondeterministicFiniteAutomaton
from scipy.sparse import dok_matrix, kron
import numpy as np

from project.build_graph import regex_to_dfa, graph_to_nfa


class AdjacencyMatrixFA:
    def __init__(
        self, finite_automaton: NondeterministicFiniteAutomaton | None = None
    ) -> None:
        if finite_automaton is None:
            self.states = set()
            self.state_to_index = {}
            self.start_states = set()
            self.final_states = set()
            self.transition_matrices = {}
            return

        self.states = finite_automaton.states
        state_list = list(self.states)
        self.state_to_index = {state: i for i, state in enumerate(state_list)}
        self.index_to_state = {i: state for i, state in enumerate(state_list)}
        self.start_states = {
            self.state_to_index[s] for s in finite_automaton.start_states
        }
        self.final_states = {
            self.state_to_index[s] for s in finite_automaton.final_states
        }

        matrices = {
            symbol: np.zeros((len(self.states), len(self.states)), dtype=bool)
            for symbol in finite_automaton.symbols
        }

        for src, dst, label in finite_automaton.to_networkx().edges(data="label"):
            if label is not None:
                symbol = Symbol(label)
                matrices[symbol][self.state_to_index[src], self.state_to_index[dst]] = (
                    True
                )

        self.transition_matrices = {
            sym: dok_matrix(mat) for sym, mat in matrices.items()
        }

    def accepts(self, word: Iterable[Symbol]) -> bool:
        current_states = self.start_states.copy()

        for symbol in word:
            if symbol not in self.transition_matrices:
                return False

            next_states = set()
            for source_state in current_states:
                _, dest_indices = self.transition_matrices[symbol][
                    source_state, :
                ].nonzero()
                next_states.update(dest_indices)

            current_states = next_states
            if not current_states:
                return False

        return bool(current_states & self.final_states)

    def get_trans_closure(self) -> np.ndarray:
        if not self.transition_matrices:
            return np.eye(len(self.states), dtype=bool)
        closure_matrix = sum(self.transition_matrices.values())
        closure_matrix.setdiag(True)
        return np.linalg.matrix_power(closure_matrix.toarray(), len(self.states))

    def is_empty(self) -> bool:
        if not self.start_states or not self.final_states:
            return True

        closure = self.get_trans_closure()
        return not any(
            closure[start_idx, final_idx]
            for start_idx in self.start_states
            for final_idx in self.final_states
        )


def intersect_automata(
    fa1: AdjacencyMatrixFA, fa2: AdjacencyMatrixFA
) -> AdjacencyMatrixFA:
    intersection = AdjacencyMatrixFA()
    for s1, s2 in product(fa1.states, fa2.states):
        combined_state = State((s1, s2))
        idx = len(fa2.states) * fa1.state_to_index[s1] + fa2.state_to_index[s2]
        intersection.states.add(combined_state)
        intersection.state_to_index[combined_state] = idx
        if (
            fa1.state_to_index[s1] in fa1.start_states
            and fa2.state_to_index[s2] in fa2.start_states
        ):
            intersection.start_states.add(idx)
        if (
            fa1.state_to_index[s1] in fa1.final_states
            and fa2.state_to_index[s2] in fa2.final_states
        ):
            intersection.final_states.add(idx)
    intersection.transition_matrices = {
        label: kron(
            fa1.transition_matrices[label], fa2.transition_matrices[label], format="dok"
        )
        for label in fa1.transition_matrices.keys() & fa2.transition_matrices.keys()
    }
    return intersection


def tensor_based_rpq(
    regex: str, graph: MultiDiGraph, start_nodes: set[int], final_nodes: set[int]
) -> set[tuple[int, int]]:
    regex_dfa = regex_to_dfa(regex)
    regex_adj = AdjacencyMatrixFA(regex_dfa)
    graph_adj = AdjacencyMatrixFA(graph_to_nfa(graph, start_nodes, final_nodes))
    intersection_fa = intersect_automata(graph_adj, regex_adj)
    closure = intersection_fa.get_trans_closure()
    result = {
        (start_node, final_node)
        for start_node in start_nodes
        for final_node in final_nodes
        if any(
            closure[
                intersection_fa.state_to_index[(start_node, regex_start)],
                intersection_fa.state_to_index[(final_node, regex_final)],
            ]
            for regex_start in regex_dfa.start_states
            for regex_final in regex_dfa.final_states
        )
    }
    return result
