from dataclasses import dataclass
from typing import Set, Tuple, Iterable
from collections import defaultdict, deque
import networkx as nx
from pyformlang.rsa import RecursiveAutomaton
from pyformlang.finite_automaton import Symbol, State


@dataclass(frozen=True)
class RSMState:
    nonterm: Symbol
    state: State


@dataclass(frozen=True)
class GSSNode:
    rsm_st: RSMState
    graph_node: int


@dataclass(frozen=True)
class Config:
    rsm_st: RSMState
    graph_node: int
    gss_node: GSSNode


def gll_based_cfpq(
    rsm: RecursiveAutomaton,
    graph: nx.DiGraph,
    start_nodes: Set[int] = None,
    final_nodes: Set[int] = None,
) -> Set[Tuple[int, int]]:
    start_nodes = start_nodes or set(graph.nodes)
    final_nodes = final_nodes or set(graph.nodes)

    graph_adj = {}
    for node in graph.nodes:
        transitions = defaultdict(set)
        for _, neighbor, label in graph.edges(node, data="label"):
            transitions[label].add(neighbor)
        graph_adj[node] = transitions

    rsm_boxes = {}
    labels_to_check = set(rsm.labels) | {rsm.initial_label}

    for label in labels_to_check:
        box = rsm.get_box(label)
        if box:
            rsm_boxes[label] = {
                "dfa": box.dfa.to_dict(),
                "start_states": set(box.start_state),
                "final_states": set(box.final_states),
            }

    gss_nodes = {}
    gss_edges = defaultdict(list)

    waiting_queue = deque()
    seen_configs = set()
    processed_configs = set()
    result_pairs = set()

    initial_gss_node = GSSNode(RSMState(Symbol("$"), State(0)), -1)

    initial_box = rsm_boxes.get(rsm.initial_label)
    if not initial_box:
        return set()

    for start_state in initial_box["start_states"]:
        for start_node in start_nodes:
            rsm_state = RSMState(rsm.initial_label, start_state)
            gss_node = GSSNode(rsm_state, start_node)

            gss_nodes.setdefault(gss_node, None)
            gss_edges[gss_node].append((initial_gss_node, rsm_state))

            config = Config(rsm_state, start_node, gss_node)
            waiting_queue.append(config)
            seen_configs.add(config)

    while waiting_queue:
        config = waiting_queue.popleft()
        if config in processed_configs:
            continue
        processed_configs.add(config)

        box_info = rsm_boxes.get(config.rsm_st.nonterm)
        if not box_info:
            continue

        transitions = box_info["dfa"]
        rsm_transitions = {}

        if config.rsm_st.state in transitions:
            for symbol, destination in transitions[config.rsm_st.state].items():
                destinations = (
                    destination
                    if isinstance(destination, Iterable)
                    and not isinstance(destination, (str, bytes))
                    else [destination]
                )
                for dest in destinations:
                    rsm_transitions.setdefault(symbol, set()).add(
                        RSMState(config.rsm_st.nonterm, dest)
                    )

        graph_transitions = graph_adj.get(config.graph_node, {})

        for label in set(graph_transitions.keys()) & set(rsm_transitions.keys()):
            neighbors = graph_transitions[label]
            for new_rsm_state in rsm_transitions[label]:
                for new_node in neighbors:
                    new_config = Config(new_rsm_state, new_node, config.gss_node)
                    if new_config not in seen_configs:
                        waiting_queue.append(new_config)
                        seen_configs.add(new_config)

        for label, target_states in rsm_transitions.items():
            if label not in rsm.labels:
                continue

            called_box = rsm_boxes.get(label)
            if not called_box:
                continue

            for start_state in called_box["start_states"]:
                call_rsm_state = RSMState(label, start_state)
                call_gss_node = GSSNode(call_rsm_state, config.graph_node)

                existing_pop_set = gss_nodes.get(call_gss_node)
                if existing_pop_set:
                    for popped_node in existing_pop_set:
                        for return_state in target_states:
                            gss_edges[call_gss_node].append(
                                (config.gss_node, return_state)
                            )
                            return_config = Config(
                                return_state, popped_node, config.gss_node
                            )
                            if return_config not in seen_configs:
                                waiting_queue.append(return_config)
                                seen_configs.add(return_config)
                    continue

                if call_gss_node not in gss_nodes:
                    gss_nodes[call_gss_node] = None

                for return_state in target_states:
                    gss_edges[call_gss_node].append((config.gss_node, return_state))

                entry_config = Config(call_rsm_state, config.graph_node, call_gss_node)
                if entry_config not in seen_configs:
                    waiting_queue.append(entry_config)
                    seen_configs.add(entry_config)

        if config.rsm_st.state in box_info["final_states"]:
            gss_nodes[config.gss_node] = gss_nodes.get(config.gss_node) or set()
            pop_set = gss_nodes[config.gss_node]
            pop_set.add(config.graph_node)

            for target_gss, return_rsm_state in gss_edges.get(config.gss_node, []):
                if target_gss == initial_gss_node:
                    result_pairs.add((config.gss_node.graph_node, config.graph_node))
                else:
                    return_config = Config(
                        return_rsm_state, config.graph_node, target_gss
                    )
                    if return_config not in seen_configs:
                        waiting_queue.append(return_config)
                        seen_configs.add(return_config)

    return {
        (start, end)
        for start, end in result_pairs
        if start in start_nodes and end in final_nodes
    }
