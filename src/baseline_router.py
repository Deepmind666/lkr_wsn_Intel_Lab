"""
Baseline routing: shortest-path to base station using geometric distance as cost.
Provides a simple, deterministic reference against ACO.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import heapq
import numpy as np


@dataclass
class Route:
    path: List[int]  # node ids with -1 representing base station
    cost: float


class BaselineRouter:
    def __init__(self):
        pass

    @staticmethod
    def _distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        return float(np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2))

    def _build_graph(self, nodes: Dict[int, Dict], topology: Dict[int, List[int]], base_station_pos: Tuple[float, float]) -> Dict[int, List[Tuple[int, float]]]:
        graph: Dict[int, List[Tuple[int, float]]] = {}
        # regular nodes edges
        for u, nbrs in topology.items():
            graph.setdefault(u, [])
            for v in nbrs:
                w = self._distance(nodes[u]['position'], nodes[v]['position'])
                graph[u].append((v, w))
        # add a virtual base station node id as -1; connect from any node within communication to BS
        graph[-1] = []
        for u, node in nodes.items():
            w = self._distance(node['position'], base_station_pos)
            # allow direct edge to BS; cost is distance
            graph.setdefault(u, []).append((-1, w))
        return graph

    def shortest_path(self, start: int, graph: Dict[int, List[Tuple[int, float]]]) -> Route:
        # Dijkstra until reaching -1 (base station)
        dist: Dict[int, float] = {start: 0.0}
        prev: Dict[int, Optional[int]] = {start: None}
        pq: List[Tuple[float, int]] = [(0.0, start)]
        visited = set()
        while pq:
            d, u = heapq.heappop(pq)
            if u in visited:
                continue
            visited.add(u)
            if u == -1:
                break
            for v, w in graph.get(u, []):
                nd = d + w
                if v not in dist or nd < dist[v]:
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(pq, (nd, v))
        if -1 not in dist:
            return Route([start], float('inf'))
        # reconstruct path
        path = []
        cur = -1
        while cur is not None:
            path.append(cur)
            cur = prev.get(cur)
        path.reverse()
        return Route(path=path, cost=dist[-1])

    def find_routes(self, cluster_heads: List[int], nodes: Dict[int, Dict], topology: Dict[int, List[int]], base_station_pos: Tuple[float, float]) -> List[Route]:
        graph = self._build_graph(nodes, topology, base_station_pos)
        routes: List[Route] = []
        for ch in cluster_heads:
            r = self.shortest_path(ch, graph)
            routes.append(r)
        return routes


