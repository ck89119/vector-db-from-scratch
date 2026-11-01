use crate::graph::{greedy_search, prune_neighbors, search_layer};
use crate::search::DeterministicRng;
use crate::{Dataset, Metric, Neighbor, Result, VectorError, VectorIndex};

#[derive(Debug, Clone, Copy)]
pub struct HnswConfig {
    pub max_connections: usize,
    pub ef_construction: usize,
    pub ef_search: usize,
    pub max_level: usize,
    pub seed: u64,
}

impl Default for HnswConfig {
    fn default() -> Self {
        Self {
            max_connections: 12,
            ef_construction: 64,
            ef_search: 40,
            max_level: 16,
            seed: 0x5eed,
        }
    }
}

#[derive(Debug, Clone)]
pub struct HnswIndex {
    dataset: Dataset,
    metric: Metric,
    config: HnswConfig,
    levels: Vec<usize>,
    layers: Vec<Vec<Vec<usize>>>,
    entry_point: usize,
    top_level: usize,
}

impl HnswIndex {
    pub fn try_new(dataset: Dataset, metric: Metric, config: HnswConfig) -> Result<Self> {
        dataset.validate_for_metric(metric)?;

        let max_conn = config.max_connections;
        if max_conn == 0
            || config.ef_construction < max_conn
            || config.ef_search == 0
            || config.max_level == 0
        {
            return Err(VectorError::InvalidConfig("Invalid config"));
        }

        let n = dataset.len();
        let mut levels = Vec::with_capacity(n);
        let mut layers: Vec<Vec<Vec<usize>>> = Vec::with_capacity(config.max_level + 1);
        let mut entry_point = 0;
        let mut top_level = 0;
        let mut rng = DeterministicRng::new(config.seed);

        for i in 0..n {
            let level = Self::random_level(&config, &mut rng);
            levels.push(level);

            if i == 0 {
                for _ in 0..=level {
                    layers.push(vec![vec![]; n]);
                }
                entry_point = 0;
                top_level = level;
                continue;
            }

            let mut cur = entry_point;
            for l in (level + 1..=top_level).rev() {
                cur = greedy_search(&dataset, metric, dataset.vector(i), &layers[l], cur, i);
            }

            let mut entry_points = vec![cur];
            for l in (0..=level.min(top_level)).rev() {
                let neighbors = search_layer(
                    &dataset,
                    metric,
                    dataset.vector(i),
                    &layers[l],
                    &entry_points,
                    config.ef_construction,
                    i,
                );

                layers[l][i] = neighbors.iter().copied().map(|n| n.row).collect::<Vec<_>>();
                prune_neighbors(&dataset, metric, i, &mut layers[l][i], max_conn);

                for v in layers[l][i].clone() {
                    layers[l][v].push(i);
                    if layers[l][v].len() > max_conn {
                        let before = layers[l][v].clone();
                        prune_neighbors(&dataset, metric, v, &mut layers[l][v], max_conn);

                        for u in before {
                            if !layers[l][v].contains(&u) {
                                layers[l][u].retain(|&x| x != v);
                            }
                        }
                    }
                }

                if !neighbors.is_empty() {
                    entry_points[0] = neighbors[0].row;
                }
            }

            if level > top_level {
                for _ in top_level + 1..=level {
                    layers.push(vec![vec![]; n]);
                }
                entry_point = i;
                top_level = level;
            }
        }

        Ok(Self {
            dataset,
            metric,
            config,
            levels,
            layers,
            entry_point,
            top_level,
        })
    }

    pub fn levels(&self) -> &[usize] {
        &self.levels
    }

    pub fn top_level(&self) -> usize {
        self.top_level
    }

    pub fn search_with_ef(
        &self,
        query: &[f32],
        k: usize,
        ef_search: usize,
    ) -> Result<Vec<Neighbor>> {
        self.dataset.validate_query(query, self.metric)?;
        if k == 0 || ef_search == 0 {
            return Err(VectorError::InvalidConfig("Invalid config"));
        }

        let mut cur = self.entry_point;
        for l in (1..=self.top_level).rev() {
            cur = greedy_search(
                &self.dataset,
                self.metric,
                query,
                &self.layers[l],
                cur,
                self.dataset.len(),
            );
        }

        let mut result = search_layer(
            &self.dataset,
            self.metric,
            query,
            &self.layers[0],
            &[cur],
            ef_search.max(k),
            self.dataset.len(),
        );
        result.truncate(k);
        Ok(result)
    }

    pub fn layer(&self, level: usize) -> Option<&[Vec<usize>]> {
        self.layers.get(level).map(Vec::as_slice)
    }

    fn random_level(config: &HnswConfig, rng: &mut DeterministicRng) -> usize {
        let mut level = 0;
        while level < config.max_level && rng.coin_flip() {
            level += 1
        }
        level
    }
}

impl VectorIndex for HnswIndex {
    fn kind(&self) -> &'static str {
        "hnsw"
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
