use crate::pq::k_means;
use crate::search::TopK;
use crate::{Dataset, Metric, Neighbor, Result, VectorError, VectorIndex};

#[derive(Debug, Clone, Copy)]
pub struct IvfFlatConfig {
    pub partitions: usize,
    pub probes: usize,
    pub iterations: usize,
    pub seed: u64,
}

impl Default for IvfFlatConfig {
    fn default() -> Self {
        Self {
            partitions: 16,
            probes: 4,
            iterations: 12,
            seed: 0x5eed,
        }
    }
}

#[derive(Debug, Clone)]
pub struct IvfFlatIndex {
    dataset: Dataset,
    metric: Metric,
    config: IvfFlatConfig,
    centroids: Vec<Vec<f32>>,
    lists: Vec<Vec<usize>>,
}

impl IvfFlatIndex {
    pub fn try_new(dataset: Dataset, metric: Metric, config: IvfFlatConfig) -> Result<Self> {
        if config.probes < 1
            || config.partitions < config.probes
            || dataset.len() < config.partitions
            || config.iterations <= 0
        {
            return Err(VectorError::InvalidConfig("Invalid config"));
        }

        dataset.validate_for_metric(metric)?;

        let (centroids, lists) = k_means(
            dataset.vectors(),
            dataset.dimension(),
            config.partitions,
            config.iterations,
            metric,
            config.seed,
        );

        Ok(Self {
            dataset,
            metric,
            config,
            centroids,
            lists,
        })
    }

    pub fn centroids(&self) -> &[Vec<f32>] {
        &self.centroids
    }

    pub fn list_sizes(&self) -> Vec<usize> {
        self.lists.iter().map(Vec::len).collect()
    }

    pub fn search_with_probes(
        &self,
        query: &[f32],
        k: usize,
        probes: usize,
    ) -> Result<Vec<Neighbor>> {
        self.dataset.validate_query(query, self.metric)?;

        if probes < 1 || probes > self.config.partitions {
            return Err(VectorError::InvalidConfig("Invalid config"));
        }

        let mut centroids = TopK::new(probes);
        for (i, c) in self.centroids.iter().enumerate() {
            centroids.push(Neighbor {
                row: i,
                distance: self.metric.distance(c, query),
            })
        }

        let mut top_k = TopK::new(k.min(self.dataset.len()));
        while let Some(c) = centroids.pop() {
            for &row in &self.lists[c.row] {
                let vec = &self.dataset.vectors()[row];
                top_k.push(Neighbor {
                    row,
                    distance: self.metric.distance(vec, query),
                })
            }
        }

        Ok(top_k.into_sorted())
    }
}

impl VectorIndex for IvfFlatIndex {
    fn kind(&self) -> &'static str {
        "ivf_flat"
    }

    fn dataset(&self) -> &Dataset {
        &self.dataset
    }

    fn metric(&self) -> Metric {
        self.metric
    }

    fn search(&self, query: &[f32], k: usize) -> Result<Vec<Neighbor>> {
        self.search_with_probes(query, k, self.config.probes)
    }
}
