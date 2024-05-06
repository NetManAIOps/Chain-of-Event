"""
Based on pool/api-level graph and events,Constructing Event-level Dependency
Graph with causality dependency for RCA


@author: hanzwang@ebay.com
"""

import copy
import networkx as nx

class DeepEventGraph:
    def __init__(self, event, G, pool_list, catalog):
        """
        :param event: event to search
        :param G: application graph
        :param deep_rule: a trained model for generating weight for edge
        """
        self.event = event
        self.G = nx.DiGraph(G)
        self.EG = nx.DiGraph()
        self.events_to_search = []
        self.pool_list = pool_list
        self.catalog = catalog

    @staticmethod
    def _sets_to_events(a_set, events, poolname):
        if isinstance(set, str):
            events.add((set, poolname))
        else:
            for e in a_set:
                events.add((e, poolname))

    def _traverse(self, G, node, events, visited):
        visited.add(node)
        if G.nodes[node]["events"]:
            self._sets_to_events(G.nodes[node]["events"], events, node)
        if G[node]:
            for n in G[node]:
                if n not in visited:
                    G.nodes[n]['pre_events'] += copy.deepcopy(events)
                    self._traverse(G, n, events, copy.deepcopy(visited))

    def get_rc_events(self):
        rc_events = set()
        for p in self.pool_list:
            for event in self.G.nodes[p]["events"]:
                if p == 'paypal' or p == 'afterpay' or p == 'adyen':
                    rc_events.add((event, p))
        # if len(rc_events) == 1 and rc_events[0][1] == 'adyen':
        #     rc_events = []
        return list(rc_events)

    def event_focus_graph(self, rc_events=None):
        for p in self.pool_list:
            for event in self.G.nodes[p]["events"]:
                self.events_to_search.append((event, p))
        if rc_events:
            "replace rc events to search"
            self.events_to_search = rc_events
        for event in self.events_to_search:
            # key = events_to_link, res = link methods
            self.searchlog = set()
            self._event_traverse((event[0], event[1]), event[1], "s")
            for n, _ in self.G.in_edges(event[1]):
                self._event_traverse((event[0], event[1]), n, "u")
            for n in self.G[event[1]]:
                self._event_traverse((event[0], event[1]), n, "d")
        return self.EG

    def _event_traverse(self, event, pool, direction):
        """
        from one event to to it's neighbors
        :param event: tuple of event and pool location
        :param pool: current target pool to search.
        :param direction: u,s,d relationships between event and pool
        :return:
        """
        log = str(event[0]) + str(event[1]) + pool + direction
        if hash(log) in self.searchlog:
            return
        # print(f"_event_traverse(self, {event}, {pool}, {direction}, DEPRULE)")
        self.searchlog.add(hash(str(event[0]) + str(event[1]) + pool + direction))
        for key in self.G.nodes[pool]["events"]:
            if pool == event[1] and key == event[0]:
                continue
            if self.catalog != 'general':
                self_event = ['DEPLOYSOFTWARE', 'ExecuteScript', 'ConfigChange', 'REFRESHCONFIG']
                upper_event = ['Badbox', 'Traffic increase', 'TPS increase']
                if event[0] in self_event or key in self_event:
                    if direction == 'd' or direction == 'u':
                        continue
                    if event[0] in self_event:
                        continue
                if event[0] in upper_event or key in upper_event:
                    if direction == 'd':
                        continue
                    if direction == 'u' and key not in upper_event:
                        continue
                else:
                    if direction == 'u':
                        continue
                res = 1
                # For events with directions check direction
                is_directional, directions = self._is_directional_event(self.EG.nodes[event]["details"])
                if not is_directional or pool in directions or event[1] in directions or direction == "s":
                    # check colo-based propagation
                    source_colo_enabled, source_colo = self. \
                        is_colo_enabled_event(self.EG.nodes[event]["details"])
                    if source_colo_enabled:
                        dest_colo_enabled, dest_colo = self. \
                            is_colo_enabled_event(self.EG.nodes[(key, pool)]["details"])
                        if dest_colo_enabled and len(source_colo & dest_colo) == 0:
                            break
                    non_rule_weight = is_directional and direction != "s"
                    self.EG.add_edge(event, (key, pool), weight=res, direction=direction, non_rule_weight=non_rule_weight)
                    self._event_traverse((key, pool), pool, "s")
                    for n, _ in self.G.in_edges(pool):
                        self._event_traverse((key, pool), n, "u")
                    for n in self.G[pool]:
                        self._event_traverse((key, pool), n, "d")
            else:
                self_end_event = ['DEPLOYSOFTWARE', 'ExecuteScript', 'ConfigChange', 'REFRESHCONFIG']
                self_noout_event = ['InternalErrorSpike', 'GC overhead', 'High JVM CPU', 'Thread usage', 'BesMarkDown']
                self_event = self_end_event + self_noout_event + ['Issue',
                                                                  # 'Latency increase',
                                                                  'WebApiErrorSpike']
                upper_event = ['Badbox', 'Traffic increase', 'TPS increase']
                if event[0] in self_event or key in self_event:
                    if direction == 'd' or direction == 'u':
                        if event[0] in self_event:
                            continue
                        if key in self_end_event or key in self_noout_event:
                            continue
                    else:
                        if event[0] in self_end_event:
                            continue

                if event[0] in upper_event or key in upper_event:
                    if direction == 'd':
                        continue
                    if direction == 'u' and key not in upper_event:
                        continue
                else:
                    if direction == 'u':
                        continue
                res = 1
                # For events with directions check direction
                is_directional, directions = self._is_directional_event(self.EG.nodes[event]["details"])
                if not is_directional or pool in directions or event[1] in directions or direction == "s":
                    # check colo-based propagation
                    source_colo_enabled, source_colo = self. \
                        is_colo_enabled_event(self.EG.nodes[event]["details"])
                    if source_colo_enabled:
                        dest_colo_enabled, dest_colo = self. \
                            is_colo_enabled_event(self.EG.nodes[(key, pool)]["details"])
                        if dest_colo_enabled and len(source_colo & dest_colo) == 0:
                            break
                    non_rule_weight = is_directional and direction != "s"
                    self.EG.add_edge(event, (key, pool), weight=res,
                                     direction=direction if key not in self_end_event else 'f',
                                     non_rule_weight=non_rule_weight)
                    self._event_traverse((key, pool), pool, "s")
                    for n, _ in self.G.in_edges(pool):
                        self._event_traverse((key, pool), n, "u")
                    for n in self.G[pool]:
                        self._event_traverse((key, pool), n, "d")

    @staticmethod
    def is_colo_enabled_event(event_detail):
        """
        :param event_detail: Check if the events have colo-based details, if yes, then
        also return the set of directions
        :return:
        is_directional
        directions
        """
        colo = set()
        for _, events_at_a_time in event_detail.items():
            for event in events_at_a_time:
                if "colo" in event.keys() and event["colo"] and event["colo"] != "UNKOWN":
                    colo.add(event["colo"])
        if len(colo) == 0:
            return False, _
        return True, colo

    @staticmethod
    def _is_directional_event(event_detail):
        """
        :param event_detail: Check if the events are all directional, if yes,
        then also return the set of directions
        :return:
        is_directional
        directions
        """
        directions = set()
        for _, events_at_a_time in event_detail.items():
            for event in events_at_a_time:
                if "target" not in event.keys() or event["target"] is None:
                    return False, _
                else:
                    directions.add(event["target"])
        return True, directions

    def _mp_search(self, start):
        events = set()
        self._sets_to_events(self.G.nodes[start[0]]["events"], events,
                             start[0])
        self._traverse(self.G, start[0], events, set())

    def add_events(self):
        for pool, events in self.event.items():
            for e, details in events.items():
                if self.G.has_node(pool):
                    # collect colo and remove duplicate
                    colo = set()
                    for _, d in details.items():
                        colo.update(set(alert.get('colo') for alert in d))
                    if (e, pool) not in self.EG:
                        self.G.nodes[pool]['events'].add(e)
                        self.EG.add_node((e, pool), colo=colo, artificial=False, details=details, type="Perfmon")
        for pool, events in self.event.items():
            for e, details in events.items():
                if self.G.has_node(pool):
                    # collect colo and remove duplicate
                    colo = set()
                    for _, d in details.items():
                        colo.update(set(alert.get('colo') for alert in d))
                        # create new entities based on event
                        for alert in d:
                            if "target" in alert.keys() and alert["target"]:
                                if alert["target"] not in self.G:
                                    # add entity
                                    self.G.add_node(alert["target"], events=set(),
                                                    pre_events=[])
                                    # create dependency
                                    self.G.add_edge(pool, alert["target"])
                                    # add events
                                elif not self.G.has_edge(pool, alert["target"]):
                                    self.G.add_edge(pool, alert["target"])
                                target_event = "Issue"
                                if alert["target"] in ['adyen', 'paypal', 'afterpay']:
                                    upper_rc_event = str.upper(alert["target"][0]) + alert["target"][1:]
                                    target_event = f'Third Party {upper_rc_event} Error'
                                if (target_event, alert["target"]) not in self.EG:
                                    self.G.nodes[alert["target"]]['events'].add(target_event)
                                    self.EG.add_node((target_event, alert["target"]),
                                                     colo=colo,
                                                     details=details,
                                                     artificial=True,
                                                     fulldetail=[details],
                                                     type="Perfmon",
                                                     trigger_type='paypal' if (alert["target"] == 'PayPal') else (
                                                         'db' if (e == 'DBMarkdown') else 'pool')
                                                     )
                                else:
                                    if 'fulldetail' in self.EG.nodes[(target_event, alert["target"])]:
                                        self.EG.nodes[(target_event, alert["target"])]['fulldetail'].append(details)
                                    else:
                                        fulldetail = [self.EG.nodes[(target_event, alert["target"])]['details'],
                                                      details]
                                        dic = {
                                            (target_event, alert["target"]): fulldetail
                                        }
                                        nx.set_node_attributes(self.EG, dic, 'fulldetail')
        for p in self.G.nodes:
            for e in list(self.G.nodes[p]['events']):
                if (e, p) not in self.EG:
                    self.G.nodes[p]['events'].remove(e)
