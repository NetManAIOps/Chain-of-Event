"""
Groot Graph Algorithms.

@author: hanzwang@ebay.com
"""

import pandas as pd
import copy
import conf.constants as const
from graph.event_graph_search import DeepEventGraph
from graph.rca_algo import rca_rank, deep_rca_rank
from sklearn import preprocessing
import numpy as np
import networkx as nx
import torch
import tensorkit.tensor as T

MAX_NUM = 100
FACTOR = 0.8
DOMAIN_RC_WEIGHT = 0.9
HITS_WEIGHT = 0.6
TRAFFIC_E = "TrafficAlert"


def deep_groot(pool_events, domain_events, subG, pool_list, domain, catalog, rca_algo=deep_rca_rank):
    domain_rc = []
    domain_rc_severity = []
    if domain_events is not None and TRAFFIC_E in domain_events and domain in domain_events[TRAFFIC_E]:
        traffic_e_region = set()
        traffic_severity = 0
        for k in domain_events[TRAFFIC_E][domain].keys():
            for d in domain_events[TRAFFIC_E][domain][k]:
                if d["country"] is not None:
                    traffic_e_region.add(d["country"])
                    if d["severity"] is not None:
                        traffic_severity = max(traffic_severity, d["severity"])
        domain_rc.append(("Traffic sudden {} in {}".format(d["alertType"], str(list(traffic_e_region))[1:-1]), domain))
        domain_rc_severity.append(traffic_severity)
    eg = DeepEventGraph(pool_events, subG, pool_list, catalog)
    # Step 4 Build Event Causal Graph
    eg.add_events()
    EG = eg.event_focus_graph()
    res = _deep_ranking(EG, eg.pool_list, rca_algo, domain_rc, domain_rc_severity,
                        limited_start_set=eg.events_to_search)
    return res, EG, eg.G


def _deep_ranking(EG, pool, rca_algo, domain_rc, domain_rc_severity, **kwargs):
    """
    Ranking Algo of Groot
    :param EG: Event Grpah with Causal links
    :param pool: Pool to start searching
    :return: Results dict
    """
    res = {}
    non_rc = []
    # if EG.number_of_edges() == 0:
    #     # domain RCA events
    #     for n in EG:
    #         if n[1] in pool:
    #             res[n[0] + "," + n[1]] = 1
    #         else:
    #             res[n[0] + "," + n[1]] = 0
    #     res = {k: v for k, v in sorted(res.items(),
    #                                    key=lambda item: - item[1])}
    #     return res
    # check if event is directional
    for n in EG:
        directional = True
        if EG.nodes[n]["artificial"]:
            continue
        for _, events_at_a_time in EG.nodes[n]["details"].items():
            for event in events_at_a_time:
                if "target" not in event.keys() or event["target"] is None:
                    directional = False
        if directional:
            non_rc.append(n)
    pr = rca_algo(EG, **kwargs)
    for p in pr:
        if isinstance(pr[p], torch.Tensor):
            pr[p] = T.to_numpy(pr[p], force_copy=True)
    # domain RCA events
    if domain_rc:
        for i in range(len(domain_rc)):
            pr[domain_rc[i]] = np.exp(domain_rc_severity[i] / 10 - 2) * 1.0 / (EG.number_of_nodes() - 1)

    result_nparra = np.array([[v] for _, v in pr.items()])
    # PR Normalization
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled_p = min_max_scaler.fit_transform(result_nparra)
    for i, p in zip(pr, np.nditer(x_scaled_p)):
        pr[i] = p.item()
    pdd = pd.DataFrame.from_dict(pr, orient='index', columns=['Score'])
    # remove no dep events (noisy)
    # if EG.number_of_edges() > 0:
    #     events_delete = list(nx.isolates(EG))
    #     for i in events_delete:
    #         pdd.at[i, 'Score'] = -1

    pdd = pdd.sort_values(by=['Score'], ascending=False)
    x = pdd.values  # returns a numpy array
    x_scaled = min_max_scaler.fit_transform(x)

    for pdr, score in zip(pdd.iterrows(), np.nditer(x_scaled)):
        res[pdr[0][0] + "," + pdr[0][1]] = np.asscalar(score)
    # for i in non_rc:
    #     res[i[0] + "," + i[1]] = 0
    return res


def basic_pool_pr(subG):
    """
    NO USE AT ALL Except:
    This is function is only for algorithm validation/benchmark for ASPLOS paper submission
    :return:
    """
    G = nx.DiGraph()
    nstart = {}
    G.add_nodes_from(subG)
    G.add_edges_from(subG.edges())
    for n in list(G.nodes()):
        if len(subG.nodes[n][const.EVENTS]) == 0:
            print(n)
            G.remove_node(n)
        else:
            nstart[n] = len(subG.nodes[n][const.EVENTS])

    result = nx.pagerank(G, nstart=nstart)
    return result


def _ranking(EG, pool, RULE, domain_rc=None, rca_algo=rca_rank):
    """
    Ranking Algo of Groot
    :param EG: Event Grpah with Causal links
    :param pool: Pool to start searching
    :return: Results dict
    """
    res = {}
    non_rc = []
    if EG.number_of_edges() == 0:
        # domain RCA events
        for n in EG:
            if n[1] in pool:
                if RULE[n[0]]["type"] == "RC":
                    domain_rc.append((n[0], n[1]))
        if domain_rc:
            for n in EG:
                if n[1] in pool:
                    res[n[0] + "," + n[1]] = 1 / (len(domain_rc) + 1)
                else:
                    res[n[0] + "," + n[1]] = 0
            for rc in domain_rc:
                res[rc[0] + "," + rc[1]] = 1
        else:
            for n in EG:
                if n[1] in pool:
                    res[n[0] + "," + n[1]] = 1
                else:
                    res[n[0] + "," + n[1]] = 0
        res = {k: v for k, v in sorted(res.items(),
                                       key=lambda item: - item[1])}
        return res
    # check if event is directional
    for n in EG:
        directional = True
        if EG.nodes[n]["artificial"]:
            continue
        for _, events_at_a_time in EG.nodes[n]["details"].items():
            for event in events_at_a_time:
                if "target" not in event.keys() or event["target"] is None:
                    directional = False
        if directional:
            non_rc.append(n)
    pr = rca_algo(EG)

    result_nparra = np.array([[v] for _, v in pr.items()])
    # PR Normalization
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled_p = min_max_scaler.fit_transform(result_nparra)
    for i, p in zip(pr, np.nditer(x_scaled_p)):
        pr[i] = round(p.item(), 3)

    pdd = pd.DataFrame.from_dict(pr, orient='index', columns=['Score'])
    # remove no dep events (noisy)
    events_delete = list(nx.isolates(EG))
    for i in events_delete:
        pdd.at[i, 'Score'] = -1
    # domain RCA events
    if domain_rc:
        dr = pd.DataFrame([DOMAIN_RC_WEIGHT] * len(domain_rc), index=domain_rc, columns=['Score'])
        pdd = pdd.append(dr)
    pdd = pdd.sort_values(by=['Score'], ascending=False)
    x = pdd.values  # returns a numpy array
    x_scaled = min_max_scaler.fit_transform(x)

    for pdr, score in zip(pdd.iterrows(), np.nditer(x_scaled)):
        res[pdr[0][0] + "," + pdr[0][1]] = round(np.asscalar(score), 3)
    for i in non_rc:
        res[i[0] + "," + i[1]] = 0
    return res


def ranking_v2(EG, pool, RULE, domain_rc=None):
    """
    Backup Version of the last tested ranking Algo of Groot
    :param EG: Event Grpah with Causal links
    :param pool: Pool to start searching
    :return: Results dict
    """
    res = {}
    if EG.number_of_edges() == 0:
        # domain RCA events
        for n in EG:
            if n[1] in pool:
                if RULE[n[0]]["type"] == "RC":
                    domain_rc.append((n[0], n[1]))
        if domain_rc:
            for n in EG:
                if n[1] in pool:
                    res[n[0] + "," + n[1]] = 1 / (len(domain_rc) + 1)
                else:
                    res[n[0] + "," + n[1]] = 0
            for rc in domain_rc:
                res[rc[0] + "," + rc[1]] = 1
        else:
            for n in EG:
                if n[1] in pool:
                    res[n[0] + "," + n[1]] = 1
                else:
                    res[n[0] + "," + n[1]] = 0
        res = {k: v for k, v in sorted(res.items(),
                                       key=lambda item: - item[1])}
        return res
    hits, _ = nx.hits(EG, max_iter=500)

    result_nparra = np.array([[v] for _, v in hits.items()])
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled_h = min_max_scaler.fit_transform(result_nparra)
    pr = nx.pagerank(EG, alpha=0.8)

    result_nparra = np.array([[v] for _, v in pr.items()])
    x_scaled_p = min_max_scaler.fit_transform(result_nparra)
    for i, p, h in zip(pr, np.nditer(x_scaled_p), np.nditer(x_scaled_h)):
        pr[i] = round(p.item(), 3) - (round(h.item(), 3) * HITS_WEIGHT)

    pdd = pd.DataFrame.from_dict(pr, orient='index', columns=['Score'])

    # remove no dep events (noisy)
    events_delete = list(nx.isolates(EG))
    for i in events_delete:
        pdd.at[i, 'Score'] = -1

    pdd = pdd.sort_values(by=['Score'], ascending=False)
    x = pdd.values  # returns a numpy array
    x_scaled = min_max_scaler.fit_transform(x)

    for pdr, score in zip(pdd.iterrows(), np.nditer(x_scaled)):
        res[pdr[0][0] + "," + pdr[0][1]] = round(np.asscalar(score), 3)

    return res


def linear_transform(input_dic, lower, upper):
    output_dic = {}
    max_num, min_num = max(input_dic.values()), min(input_dic.values())
    origin_len, new_len = (1 if max_num == min_num else max_num - min_num), upper - lower
    for k, v in input_dic.items():
        output_dic[k] = round(upper - FACTOR * new_len * (max_num - v) / origin_len, 4)
    return {k: v for k, v in sorted(output_dic.items(), key=lambda item: item[1], reverse=True)}