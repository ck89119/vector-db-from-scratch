use crate::graph::{prune_neighbors, search_layer};
use crate::{Dataset, Metric, Neighbor, Result, VectorError, VectorIndex};

#[derive(Debug, Clone, Copy)]
pub struct NswConfig {
    pub max_connections: usize,
    pub ef_construction: usize,
    pub ef_search: usize,
}

impl Default for NswConfig {
    fn default() -> Self {
        Self {
            max_connections: 12,
            ef_construction: 48,
            ef_search: 32,
        }
    }
}

#[derive(Debug, Clone)]
pub struct NswIndex {
    dataset: Dataset,
    metric: Metric,
    config: NswConfig,
    adjacency: Vec<Vec<usize>>,
    entry_point: usize,
}

impl NswIndex {
    pub fn try_new(dataset: Dataset, metric: Metric, config: NswConfig) -> Result<Self> {
        dataset.validate_for_metric(metric)?;

        let max_conn = config.max_connections;
        if !(max_conn > 0 && config.ef_construction >= max_conn && config.ef_search > 0) {
            return Err(VectorError::InvalidConfig("Invalid config"));
        }

        let mut adjacency = vec![vec![]; dataset.len()];
        let mut entry_points = vec![0];
        for r in 1..dataset.len() {
            let neighbors = search_layer(
                &dataset,
                metric,
                dataset.vector(r),
                adjacency.as_ref(),
                entry_points.as_ref(),
                config.ef_construction,
                r,
            );
            adjacency[r] = neighbors.iter().copied().map(|n| n.row).collect::<Vec<_>>();
            prune_neighbors(&dataset, metric, r, &mut adjacency[r], max_conn);

            for v in adjacency[r].clone() {
                adjacency[v].push(r);
                if adjacency[v].len() > max_conn {
                    let before = adjacency[v].clone();
                    prune_neighbors(&dataset, metric, v, &mut adjacency[v], max_conn);

                    for u in before {
                        if !adjacency[v].contains(&u) {
                            adjacency[u].retain(|&x| x != v);
                        }
                    }
                }
            }

            entry_points[0] = r;
        }

        Ok(Self {
            dataset,
            metric,
            config,
            adjacency,
            entry_point: entry_points[0],
        })
    }

    pub fn adjacency(&self) -> &[Vec<usize>] {
        &self.adjacency
    }

    pub fn search_with_ef(
        &self,
        query: &[f32],
        k: usize,
        ef_search: usize,
    ) -> Result<Vec<Neighbor>> {
        self.dataset.validate_query(query, self.metric)?;
        if ef_search == 0 {
            return Err(VectorError::InvalidConfig("Invalid config"));
        }

        let mut neighbors = search_layer(
            &self.dataset,
            self.metric,
            query,
            self.adjacency(),
            vec![self.entry_point].as_ref(),
            ef_search.max(k),
            self.dataset.len(),
        );
        neighbors.resize(
            k,
            Neighbor {
                row: 0,
                distance: 0.0,
            },
        );
        Ok(neighbors)
    }
}

impl VectorIndex for NswIndex {
    fn kind(&self) -> &'static str {
        "nsw"
    }

    fn dataset(&self) -> &Dataset {
        &self.dataset
    }

    fn metric(&self) -> Metric {
        self.metric
    }

    fn search(&self, query: &[f32], k: usize) -> Result<Vec<Neighbor>> {
        self.search_with_ef(query, k, self.config.ef_search)
    }
}
