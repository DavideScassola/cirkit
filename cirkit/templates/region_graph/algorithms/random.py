import itertools
from collections import defaultdict
from collections.abc import Sequence

import numpy as np

from cirkit.templates.region_graph.graph import (
    PartitionNode,
    RegionGraph,
    RegionGraphNode,
    RegionNode,
)
from cirkit.utils.scope import Scope

def RandomBinaryTree(
    num_variables: int,
    *,
    depth: int | None = None,
    num_repetitions: int = 1,
    min_leaf_size: int = 4,
    seed: int = 42,
) -> RegionGraph:
    from collections import defaultdict
    import numpy as np

    if num_variables <= 0:
        raise ValueError("The number of variables must be positive")
    if num_repetitions <= 0:
        raise ValueError("The number of repetitions must be positive")
    max_depth = int(np.ceil(np.log2(num_variables)))
    if depth is None:
        depth = max_depth
    elif depth < 0 or depth > max_depth:
        raise ValueError(f"The depth must be between 0 and {max_depth}")
    if min_leaf_size < 1:
        raise ValueError("min_leaf_size must be at least 1")

    random_state = np.random.RandomState(seed)
    root = RegionNode(range(num_variables))
    nodes = [root]
    in_nodes = defaultdict(list)

    def random_scope_partitioning(scope, num_parts=2):
        scope = list(scope)
        random_state.shuffle(scope)
        split_point = [0, len(scope) // 2, len(scope)]
        return [Scope(scope[l:r]) for l, r in zip(split_point[:-1], split_point[1:]) if r - l >= min_leaf_size]

    for _ in range(num_repetitions):
        frontier = [root]
        for _ in range(depth):
            next_frontier = []
            for rgn in frontier:
                if len(rgn.scope) <= min_leaf_size:
                    continue  # Don't split small scopes
                scopes = random_scope_partitioning(rgn.scope, num_parts=2)
                if len(scopes) < 2:
                    continue  # Cannot partition further
                partition_node = PartitionNode(rgn.scope)
                region_nodes = [RegionNode(scope) for scope in scopes]
                nodes.append(partition_node)
                nodes.extend(region_nodes)
                in_nodes[rgn].append(partition_node)
                in_nodes[partition_node] = region_nodes
                next_frontier.extend(region_nodes)
            frontier = next_frontier

    return RegionGraph(nodes, in_nodes, outputs=[root])
