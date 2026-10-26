use crate::{Dataset, Metric, Neighbor};
use std::collections::{BinaryHeap, HashSet};

pub(crate) fn search_layer(
    dataset: &Dataset,
    metric: Metric,
    query: &[f32],
    adjacency: &[Vec<usize>],
    entry_points: &[usize],
    ef: usize,
    allowed_rows: usize,
) -> Vec<Neighbor> {
    if ef == 0 {
        return vec![];
    }

    let mut candidates = BinaryHeap::with_capacity(ef);
    let mut results = BinaryHeap::with_capacity(ef + 1);
    let mut visited = HashSet::new();

    for &row in entry_points {
        if row >= allowed_rows {
            continue;
        }
        visited.insert(row);
    }
    for &row in &visited {
        let d = metric.distance(query, dataset.vector(row));
        candidates.push(Neighbor { row, distance: -d });
        results.push(Neighbor { row, distance: d });
        if results.len() > ef {
            results.pop();
        }
    }

    while let Some(candidate) = candidates.pop() {
        if results.len() == ef && -candidate.distance > results.peek().unwrap().distance {
            break;
        }

        for &v in &adjacency[candidate.row] {
            if v >= allowed_rows || visited.contains(&v) {
                continue;
            }
            visited.insert(v);

            let d = metric.distance(query, dataset.vector(v));
            let neighbor = Neighbor {
                row: v,
                distance: d,
            };
            if results.len() < ef || neighbor.lt(results.peek().unwrap()) {
                candidates.push(Neighbor {
                    row: v,
                    distance: -d,
                });
                results.push(neighbor);
                if results.len() > ef {
                    results.pop();
                }
            }
        }
    }

    results.into_sorted_vec()
}

pub(crate) fn greedy_search(
    dataset: &Dataset,
    metric: Metric,
    query: &[f32],
    adjacency: &[Vec<usize>],
    entry: usize,
    allowed_rows: usize,
) -> usize {
    let mut cur = Neighbor {
        row: entry,
        distance: metric.distance(query, dataset.vector(entry)),
    };
    loop {
        if let Some(neighbor) = adjacency[cur.row]
            .iter()
            .filter(|&&row| row < allowed_rows)
            .map(|&row| Neighbor {
                row,
                distance: metric.distance(query, dataset.vector(row)),
            })
            .min()
            && neighbor.lt(&cur)
        {
            cur = neighbor;
        } else {
            break;
        }
    }

    cur.row
}

pub(crate) fn prune_neighbors(
    dataset: &Dataset,
    metric: Metric,
    owner: usize,
    neighbors: &mut Vec<usize>,
    max_connections: usize,
) {
    let mut heap = BinaryHeap::with_capacity(max_connections + 1);
    let vectors = dataset.vectors();

    let neighbor_set: HashSet<usize> = neighbors.iter().copied().collect();
    for row in neighbor_set {
        let distance = metric.distance(vectors[owner].as_ref(), vectors[row].as_ref());
        heap.push(Neighbor { row, distance });
        if heap.len() > max_connections {
            heap.pop();
        }
    }

    *neighbors = heap.into_sorted_vec().iter().map(|n| n.row).collect();
}
