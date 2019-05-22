use crate::data_frame::*;
use crate::learner::*;
use crate::utils::numeric;
use log::*;
use rand::seq::SliceRandom;
use serde::{Serialize, Deserialize};

#[derive(Clone, Serialize, Deserialize)]
pub struct RandomForest<L> {
    /// The base learner
    pub tree: L,
    /// The trained learners
    pub learners: Vec<L>,
    /// The number of learners to be trained
    pub n_estimators: usize,
    /// The fraction of samples to train fir individual tree
    pub sub_sample: f64,
}

#[derive(Clone, Serialize, Deserialize)]
pub enum RandomForestConfig {
    NEstimators(usize),
    SubSample(f64),
}

impl<L: Learner + Clone + Send + Sync> RandomForest<L> {
    pub fn new(base_learner: L) -> Self {
        Self {
            tree: base_learner,
            learners: vec![],
            n_estimators: 100,
            sub_sample: 1.0,
        }
    }
    pub fn from_configs(base_learner: L, configs: Vec<RandomForestConfig>) -> Self {
        let mut forest = RandomForest::new(base_learner);
        for config in configs {
            match config {
                RandomForestConfig::NEstimators(e) => forest.n_estimators = e,
                RandomForestConfig::SubSample(s) => forest.sub_sample = s,
            }
        }
        forest
    }

    /// Returns the sub sample x, residual and feature orders
    fn choose_subsample(&self, x: &DataFrame, y: &DataFrame) -> (DataFrame, DataFrame) {
        let mut rng = rand::thread_rng();
        let mut orders: Vec<usize> = (0..x.rows()).collect();
        orders.shuffle(&mut rng);
        let sub_sample_size = (self.sub_sample * orders.len() as f64) as usize;
        orders.resize(sub_sample_size, 0);

        let mut buffer = Vec::with_capacity(sub_sample_size);
        let mut y_buffer = Vec::with_capacity(sub_sample_size);
        for index in orders {
            buffer.extend(x.row(index).iter());
            y_buffer.push(y[[0, index]]);
        }
        (
            DataFrame::from_shape_vec((sub_sample_size, x.cols()), buffer).unwrap(),
            DataFrame::from_shape_vec((1, sub_sample_size), y_buffer).unwrap(),
        )
    }
}

impl<L: Learner + Clone + Send + Sync> Learner for RandomForest<L> {
    fn fit(&mut self, x: &DataFrame, y: &DataFrame) {
        // train all trees in parallel
        let new_learners: Vec<L> = (0..self.n_estimators)
            .into_iter()
            .map(|_n| {
                let (sub_x, sub_y) = self.choose_subsample(&x, &y);
                let mut tree = self.tree.clone();
                tree.fit(&sub_x, &sub_y);
                let pred = tree.predict(&x);
                info!("score at step {}: {}", self.learners.len() + _n, numeric::r2_score(&y, &pred));

                tree
            })
            .collect();
        self.learners.extend(new_learners);
    }

    fn predict(&self, df: &DataFrame) -> DataFrame {
        if self.learners.is_empty() {
            panic!("Random Forest is not trained");
        }
        let mut result_sum =
            DataFrame::from_shape_vec((1, df.rows()), vec![0.0; df.rows()]).unwrap();

        for l in &self.learners {
            result_sum = result_sum + &l.predict(&df);
        }

        result_sum / self.learners.len() as V
    }
}
