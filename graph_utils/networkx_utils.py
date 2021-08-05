#!/usr/bin/env python3

import dgl, torch
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


def networkx2dgl(nx_graph: nx.MultiDiGraph, 
        node2id: Dict[str, int], 
        node2type: Dict[str, str], 
        etype2id: Dict[str, int], 
        node_type_range: Dict[str, list],
        **kwargs) -> dgl.DGLHeteroGraph:
    hetero_edges = dict()
    node2id_ = {t: dict() for t in node_type_range}
    for n, i in node2id.items():
        t = node2type[n]
        nid = node2id[n] - node_type_range[t][0]
        node2id_[t][n] = nid
    node2id = node2id_
    for a, b, r in nx_graph.edges:
        etype = nx_graph[a][b][r]["etype"]
        ta, tb = node2type[a], node2type[b]
        rel_type = (ta, etype, tb)
        if rel_type not in hetero_edges:
            hetero_edges[rel_type] = [list(), list()]
        hetero_edges[rel_type][0].append(node2id[ta][a])
        hetero_edges[rel_type][1].append(node2id[tb][b])
    for rel_type in hetero_edges:
        src, dst = hetero_edges[rel_type]
        hetero_edges[rel_type] = (torch.as_tensor(src, dtype=torch.long), torch.as_tensor(dst, dtype=torch.long))
    id2node = {t: dict() for t in node2id}
    for t in node2id:
        id2node[t] = {i: n for n, i in node2id[t].items()}
    return dgl.heterograph(hetero_edges), node2id, id2node


