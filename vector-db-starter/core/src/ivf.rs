use crate::search::{DeterministicRng, TopK};
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

        let mut rng = DeterministicRng::new(config.seed);
        let mut offsets: Vec<usize> = (0..dataset.len()).collect();
        rng.shuffle(offsets.as_mut_slice());

        let mut lists = (0..config.partitions)
            .map(|i| vec![offsets[i]])
            .collect::<Vec<_>>();
        let mut centroids = vec![vec![]; lists.len()];

        for _ in 0..config.iterations {
            let dim = dataset.dimension();
            let normalize = |vec: &mut Vec<f32>| {
                let norm = Metric::squared_norm(vec).sqrt();
                for x in vec {
                    *x = ((*x as f64) / norm) as f32;
                }
            };

            for (idx, row_list) in lists.iter().enumerate() {
                if row_list.is_empty() {
                    continue;
                }

                let mut sums = vec![0f64; dim];
                for &row in row_list {
                    let vec = &dataset.vectors()[row];
                    for j in 0..dim {
                        sums[j] += vec[j] as f64;
                    }
                }

                let mut centroid = sums
                    .iter()
                    .map(|sum| (sum / row_list.len() as f64) as f32)
                    .collect::<Vec<_>>();

                if metric == Metric::Cosine {
                    if Metric::squared_norm(&centroid) == 0.0 {
                        centroid = dataset.vectors()[row_list[0]].to_vec();
                    }
                    normalize(&mut centroid);
                }

                centroids[idx] = centroid;
            }

            for (idx, row_list) in lists.iter().enumerate() {
                if !row_list.is_empty() {
                    continue;
                }

                let mut centroid = dataset
                    .vectors()
                    .iter()
                    .map(|row| {
                        let dis = centroids
                            .iter()
                            .map(|c| metric.distance(row, c))
                            .min_by(|a, b| a.total_cmp(&b))
                            .expect("empty centroids");
                        (row, dis)
                    })
                    .max_by(|a, b| a.1.total_cmp(&b.1))
                    .expect("empty dataset")
                    .0
                    .clone();

                if metric == Metric::Cosine {
                    normalize(&mut centroid);
                }
                centroids[idx] = centroid;
            }

            let mut new_lists = vec![vec![]; centroids.len()];
            for &row in &offsets {
                let centroid_idx = centroids
                    .iter()
                    .enumerate()
                    .map(|(i, c)| (i, metric.distance(&dataset.vectors()[row], c)))
                    .min_by(|a, b| a.1.total_cmp(&b.1))
                    .expect("empty centroids")
                    .0;
                new_lists[centroid_idx].push(row);
            }

            if lists.eq(&new_lists) {
                break;
            }
            lists = new_lists;
        }

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

        let mut centroids = self
            .centroids
            .iter()
            .enumerate()
            .map(|(i, c)| (i, self.metric.distance(c, query)))
            .collect::<Vec<_>>();
        centroids.sort_unstable_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));

        let mut top_k = TopK::new(k.min(self.dataset.len()));
        for i in 0..probes {
            for &row in &self.lists[centroids[i].0] {
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
