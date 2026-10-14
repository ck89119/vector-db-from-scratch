use std::sync::Arc;

use crate::{Metric, Result, VectorError};

#[derive(Debug, Clone)]
pub struct Dataset {
    vectors: Arc<[Vec<f32>]>,
    dimension: usize,
}

impl Dataset {
    pub fn try_new(vectors: Vec<Vec<f32>>) -> Result<Self> {
        if vectors.is_empty() {
            return Err(VectorError::EmptyDataset);
        }

        let dim = vectors[0].len();
        if dim == 0 {
            return Err(VectorError::EmptyVector);
        }

        for (i, vec) in vectors.iter().enumerate() {
            if vec.len() != dim {
                return Err(VectorError::DimensionMismatch {
                    expected: dim,
                    actual: vec.len(),
                });
            }

            for (j, x) in vec.iter().enumerate() {
                if !x.is_finite() {
                    return Err(VectorError::NonFiniteValue {
                        vector: i,
                        dimension: j,
                    });
                }
            }
        }

        Ok(Dataset {
            vectors: Arc::from(vectors),
            dimension: dim,
        })
    }

    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    pub fn dimension(&self) -> usize {
        self.dimension
    }

    pub fn vector(&self, row: usize) -> &[f32] {
        &self.vectors[row]
    }

    pub fn vectors(&self) -> &[Vec<f32>] {
        &self.vectors
    }

    pub(crate) fn validate_for_metric(&self, metric: Metric) -> Result<()> {
        if metric != Metric::Cosine {
            return Ok(());
        }

        for (i, vec) in self.vectors.iter().enumerate() {
            if vec.iter().all(|&x| x == 0f32) {
                return Err(VectorError::ZeroNorm { vector: i });
            }
        }

        Ok(())
    }

    pub(crate) fn validate_query(&self, query: &[f32], metric: Metric) -> Result<()> {
        if query.len() != self.dimension {
            return Err(VectorError::DimensionMismatch {
                expected: self.dimension,
                actual: query.len(),
            });
        }

        for (j, x) in query.iter().enumerate() {
            if !x.is_finite() {
                return Err(VectorError::NonFiniteValue {
                    vector: self.vectors.len(),
                    dimension: j,
                });
            }
        }

        if metric == Metric::Cosine {
            if query.iter().all(|&x| x == 0f32) {
                return Err(VectorError::ZeroNorm {
                    vector: self.vectors.len(),
                });
            }
        }

        Ok(())
    }
}
