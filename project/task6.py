from pyformlang.cfg import Production, Terminal, CFG, Epsilon
from networkx import DiGraph
from collections import defaultdict


def cfg_to_weak_normal_form(cfg: CFG) -> CFG:
    nullable = cfg.get_nullable_symbols()
    cfg = cfg.to_normal_form()
    new_productions = cfg.productions | {
        Production(nonterm, [Epsilon()]) for nonterm in nullable
    }
    return CFG(
        cfg.variables, cfg.terminals, cfg.start_symbol, new_productions
    ).remove_useless_symbols()


def hellings_based_cfpq(
    cfg: CFG,
    graph: DiGraph,
    start_nodes: set[int] = None,
    final_nodes: set[int] = None,
) -> set[tuple[int, int]]:
    cfg_wnf = cfg_to_weak_normal_form(cfg)
    start_nodes = start_nodes or set(graph.nodes)
    final_nodes = final_nodes or set(graph.nodes)

    R = set()
    term_productions = defaultdict(list)
    binary_productions = []

    for production in cfg_wnf.productions:
        body = production.body
        if len(body) == 1 and isinstance(body[0], Terminal):
            term_productions[body[0].value].append(production.head)
        elif len(body) == 2:
            binary_productions.append(production)

    for u, v, label in graph.edges(data="label"):
        for head in term_productions.get(label, []):
            R.add((u, head, v))

    for node in graph.nodes:
        for head in cfg_wnf.get_nullable_symbols():
            R.add((node, head, node))

    changed = True
    while changed:
        changed = False
        new_edges = set()

        for u, B, v in R:
            for v2, C, w in R:
                if v == v2:
                    for production in binary_productions:
                        if (
                            production.body == [B, C]
                            and (u, production.head, w) not in R
                        ):
                            new_edges.add((u, production.head, w))

        if new_edges:
            changed = True
            R |= new_edges

    return {
        (u, w)
        for u, head, w in R
        if head == cfg_wnf.start_symbol and u in start_nodes and w in final_nodes
    }
