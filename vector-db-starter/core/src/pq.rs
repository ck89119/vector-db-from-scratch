use crate::search::{DeterministicRng, TopK};
use crate::{Dataset, Metric, Neighbor, Result, VectorError, VectorIndex};
use std::mem::size_of;

#[derive(Debug, Clone, Copy)]
pub struct IvfPqConfig {
    pub partitions: usize,
    pub probes: usize,
    pub iterations: usize,
    pub subquantizers: usize,
    pub codebook_size: usize,
    pub rerank: usize,
    pub seed: u64,
}

impl Default for IvfPqConfig {
    fn default() -> Self {
        Self {
            partitions: 16,
            probes: 4,
            iterations: 12,
            subquantizers: 4,
            codebook_size: 16,
            rerank: 32,
            seed: 0x5eed,
        }
    }
}

#[derive(Debug, Clone)]
struct QuantizedRow {
    row: usize,
    codes: Vec<u8>,
}

#[derive(Debug, Clone)]
pub struct IvfPqIndex {
    dataset: Dataset,
    config: IvfPqConfig,
    centroids: Vec<Vec<f32>>,
    codebooks: Vec<Vec<Vec<f32>>>,
    lists: Vec<Vec<QuantizedRow>>,
}

impl IvfPqIndex {
    pub fn try_new(dataset: Dataset, metric: Metric, config: IvfPqConfig) -> Result<Self> {
        let dim = dataset.dimension();
        dataset.validate_for_metric(metric)?;

        if metric != Metric::Euclidean {
            return Err(VectorError::InvalidConfig("Invalid config"));
        }

        if config.probes < 1
            || config.partitions < config.probes
            || config.partitions > dataset.len()
            || config.iterations == 0
        {
            return Err(VectorError::InvalidConfig("Invalid config"));
        }

        if config.subquantizers == 0 || dim % config.subquantizers != 0 {
            return Err(VectorError::InvalidConfig("Invalid config"));
        }

        if config.codebook_size < 2 || config.codebook_size > 256.min(dataset.len()) {
            return Err(VectorError::InvalidConfig("Invalid config"));
        }

        if config.rerank == 0 {
            return Err(VectorError::InvalidConfig("Invalid config"));
        }

        let mut seed_rng = DeterministicRng::new(config.seed);
        let (centroids, lists) = k_means(
            dataset.vectors(),
            dim,
            config.partitions,
            config.iterations,
            metric,
            seed_rng.next_u64(),
        );

        let mut residual = dataset.vectors().to_vec();
        for i in 0..centroids.len() {
            for &row in &lists[i] {
                for k in 0..dim {
                    residual[row][k] -= centroids[i][k];
                    if !residual[row][k].is_finite() {
                        return Err(VectorError::InvalidConfig(
                            "IVF-PQ training residuals must remain finite",
                        ));
                    }
                }
            }
        }

        let mut codebooks = Vec::with_capacity(config.subquantizers);
        let mut lists = lists
            .iter()
            .map(|list| {
                list.iter()
                    .map(|&row| QuantizedRow {
                        row,
                        codes: vec![0; config.subquantizers],
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let mut row_to_lists = vec![(0, 0); dataset.len()];
        for i in 0..centroids.len() {
            for j in 0..lists[i].len() {
                row_to_lists[lists[i][j].row] = (i, j);
            }
        }

        let sub_dim = dim / config.subquantizers;
        for k in 0..config.subquantizers {
            let offset = sub_dim * k;
            let sub_dataset = residual
                .iter()
                .map(|v| v[offset..offset + sub_dim].to_vec())
                .collect::<Vec<_>>();

            let (codebook, codebook_lists) = k_means(
                sub_dataset.as_slice(),
                sub_dim,
                config.codebook_size,
                config.iterations,
                metric,
                seed_rng.next_u64(),
            );

            for code in 0..codebook.len() {
                for &row in &codebook_lists[code] {
                    let (i, j) = row_to_lists[row];
                    lists[i][j].codes[k] = code as u8;
                }
            }

            codebooks.push(codebook);
        }

        Ok(Self {
            dataset,
            config,
            centroids,
            codebooks,
            lists,
        })
    }

    pub fn centroids(&self) -> &[Vec<f32>] {
        &self.centroids
    }

    pub fn codebooks(&self) -> &[Vec<Vec<f32>>] {
        &self.codebooks
    }

    pub fn list_sizes(&self) -> Vec<usize> {
        self.lists.iter().map(Vec::len).collect()
    }

    pub fn encoded_bytes(&self) -> usize {
        self.lists.iter().flatten().map(|row| row.codes.len()).sum()
    }

    pub fn codebook_bytes(&self) -> usize {
        self.codebooks
            .iter()
            .flatten()
            .map(|centroid| centroid.len() * size_of::<f32>())
            .sum()
    }

    pub fn full_precision_bytes(&self) -> usize {
        self.dataset.len() * self.dataset.dimension() * size_of::<f32>()
    }

    pub fn search_with_probes(
        &self,
        query: &[f32],
        k: usize,
        probes: usize,
        rerank: usize,
    ) -> Result<Vec<Neighbor>> {
        self.dataset.validate_query(query, self.metric())?;
        if probes < 1 || probes > self.config.partitions {
            return Err(VectorError::InvalidConfig("Invalid config"));
        }
        if rerank == 0 {
            return Err(VectorError::InvalidConfig("Invalid config"));
        }

        let mut centroids = TopK::new(probes);
        for (i, c) in self.centroids.iter().enumerate() {
            centroids.push(Neighbor {
                row: i,
                distance: self.metric().distance(c, query),
            })
        }

        let mut rerank = TopK::new(rerank.max(k));
        let sub_dim = self.dataset.dimension() / self.config.subquantizers;
        while let Some(c) = centroids.pop() {
            let residual = query
                .iter()
                .zip(&self.centroids[c.row])
                .map(|(a, b)| {
                    let val = a - b;
                    if val.is_finite() {
                        Ok(val)
                    } else {
                        Err(VectorError::InvalidConfig(
                            "IVF-PQ training residuals must remain finite",
                        ))
                    }
                })
                .collect::<Result<Vec<_>>>()?;

            for qr in &self.lists[c.row] {
                let mut sum = 0f64;
                for k in 0..self.config.subquantizers {
                    let offset = sub_dim * k;
                    let codebook = &self.codebooks[k];
                    sum += self.metric().distance(
                        &residual[offset..offset + sub_dim],
                        &codebook[qr.codes[k] as usize],
                    ) as f64;
                }

                rerank.push(Neighbor {
                    row: qr.row,
                    distance: sum as f32,
                })
            }
        }

        let mut top_k = TopK::new(k);
        while let Some(vec) = rerank.pop() {
            let distance = self.metric().distance(query, self.dataset.vector(vec.row));
            if !distance.is_finite() {
                continue;
            }
            top_k.push(Neighbor {
                row: vec.row,
                distance,
            });
        }

        if top_k.len() < k {
            Err(VectorError::InvalidConfig(
                "IVF-PQ result distances must remain representable as finite f32",
            ))
        } else {
            Ok(top_k.into_sorted())
        }
    }
}

impl VectorIndex for IvfPqIndex {
    fn kind(&self) -> &'static str {
        "ivf_pq"
    }

    fn dataset(&self) -> &Dataset {
        &self.dataset
    }

    fn metric(&self) -> Metric {
        Metric::Euclidean
    }

    fn search(&self, query: &[f32], k: usize) -> Result<Vec<Neighbor>> {
        self.search_with_probes(query, k, self.config.probes, self.config.rerank)
    }
}

pub fn k_means<T: AsRef<[f32]>>(
    vectors: &[T],
    dim: usize,
    partitions: usize,
    iterations: usize,
    metric: Metric,
    seed: u64,
) -> (Vec<Vec<f32>>, Vec<Vec<usize>>) {
    let mut rng = DeterministicRng::new(seed);
    let mut offsets: Vec<usize> = (0..vectors.len()).collect();
    rng.shuffle(offsets.as_mut_slice());

    let mut lists = (0..partitions)
        .map(|i| vec![offsets[i]])
        .collect::<Vec<_>>();
    let mut centroids = vec![vec![]; lists.len()];

    for _ in 0..iterations {
        let normalize = |vec: &mut [f32]| {
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
                let vec = vectors[row].as_ref();
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
                    centroid = vectors[row_list[0]].as_ref().to_vec();
                }
                normalize(&mut centroid);
            }

            centroids[idx] = centroid;
        }

        for (idx, row_list) in lists.iter().enumerate() {
            if !row_list.is_empty() {
                continue;
            }

            let mut centroid = vectors
                .iter()
                .map(|row| {
                    let dis = centroids
                        .iter()
                        .map(|c| metric.distance(row.as_ref(), c))
                        .min_by(|a, b| a.total_cmp(&b))
                        .expect("empty centroids");
                    (row.as_ref(), dis)
                })
                .max_by(|a, b| a.1.total_cmp(&b.1))
                .expect("empty dataset")
                .0
                .to_vec();

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
                .map(|(i, c)| (i, metric.distance(vectors[row].as_ref(), c)))
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

    (centroids, lists)
}
