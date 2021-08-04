#!/usr/bin/env python3

import networkx as nx
from collections import defaultdict, OrderedDict

def load_graph_as_networkx(links, add_self_loop: bool, link_columns=5):
    if link_columns == 5:
        return load_graph_as_networkx_5_colums(links, add_self_loop)

def load_graph_as_networkx_5_colums(links, add_self_loop: bool, link_columns=5):
    id_mapping = {
            "node2id": OrderedDict(),
            "id2node": OrderedDict(),
            "node2type": OrderedDict() ,
            "etype2id": OrderedDict(),
            "node_type_range": dict()
        }

    type_nodes = defaultdict(set)
    etypes = set()
    if add_self_loop:
        etypes.add("__self__")

    if type(links) is str:
        links = [links]
    graph = nx.MultiDiGraph()
    seen_nodes = set()
    for fn in links:
        with open(fn) as infile:
            for l in infile:
                ta, a, r, tb, b = l.strip().split('\t')[0:5]
                etypes.add(r)
                for n in [a, b]:
                    if add_self_loop and n not in seen_nodes:
                        graph.add_edge(n, n, etype="__self__")
                    seen_nodes.add(n)
                type_nodes[ta].add(a)
                type_nodes[tb].add(b)
                graph.add_edge(a, b, etype=r)

    acc_cnt = 0
    for t in sorted(list(type_nodes.keys())):
        id_mapping["node_type_range"][t] = [
                acc_cnt,
                acc_cnt + len(type_nodes[t])
            ]
        for i, n in enumerate(sorted(list(type_nodes[t]))):
            id_mapping["node2id"][n] = acc_cnt + i
            id_mapping["node2type"][n] = t
        acc_cnt += len(type_nodes[t])

    for i, e in enumerate(sorted(list(etypes))):
        id_mapping["etype2id"][e] = i

    for n, i in id_mapping["node2id"].items():
        id_mapping["id2node"][i] = n

    return graph, id_mapping

