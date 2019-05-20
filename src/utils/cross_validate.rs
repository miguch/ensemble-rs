use crate::data_frame::*;
use crate::learner::Learner;
use log::*;
use rand::prelude::*;

use std::time;

pub struct KFold {
    pub splits: usize,
}

#[derive(Debug)]
pub struct CrossValidateScore {
    pub train_time: Vec<u128>,
    pub predict_time: Vec<u128>,
    pub train_score: Vec<V>,
    pub validation_score: Vec<V>,
}

impl KFold {
    pub fn new(k: usize) -> Self {
        Self { splits: k }
    }

    pub fn cross_validate<L, M>(
        &self,
        learner: L,
        x: &DataFrame,
        y: &DataFrame,
        metric: M,
    ) -> CrossValidateScore
    where
        L: Learner + Clone,
        M: Fn(&DataFrame, &DataFrame) -> V,
    {
        let samples = x.rows();
        let fold_size = samples / self.splits;
        let mut rng = rand::thread_rng();
        // Shuffle sample order
        let mut sample_orders: Vec<usize> = (0..samples).into_iter().collect();
        sample_orders.shuffle(&mut rng);
        let mut results = CrossValidateScore {
            train_time: vec![],
            predict_time: vec![],
            train_score: vec![],
            validation_score: vec![],
        };

        for i in 0..self.splits {
            let test_range = (fold_size * i, fold_size * (i + 1));
            let mut train_df = Vec::new();
            let mut train_labels = vec![];
            let mut test_df = Vec::new();
            let mut test_labels = vec![];
            for k in 0..samples {
                let mut row = vec![];
                let k_sample = sample_orders[k];
                for j in 0..x.cols() {
                    row.push(x[[k_sample, j]]);
                }
                if k < test_range.1 && k >= test_range.0 {
                    // Validation set
                    test_df.extend(row);
                    test_labels.push(y[[0, k_sample]])
                } else {
                    // Training set
                    train_df.extend(row);
                    train_labels.push(y[[0, k_sample]])
                }
            }

            let train_df =
                DataFrame::from_shape_vec((samples - fold_size, x.cols()), train_df).unwrap();
            let test_df = DataFrame::from_shape_vec((fold_size, x.cols()), test_df).unwrap();
            let train_labels =
                DataFrame::from_shape_vec((1, samples - fold_size), train_labels).unwrap();
            let test_labels = DataFrame::from_shape_vec((1, fold_size), test_labels).unwrap();

            let start = time::SystemTime::now();
            let mut model = learner.clone();
            model.fit(&train_df, &train_labels);
            results
                .train_time
                .push(start.elapsed().unwrap().as_millis());

            let train_pred = model.predict(&train_df);
            results.train_score.push(metric(&train_labels, &train_pred));

            let start = time::SystemTime::now();
            let test_pred = model.predict(&test_df);
            results
                .validation_score
                .push(metric(&test_labels, &test_pred));
            results
                .predict_time
                .push(start.elapsed().unwrap().as_millis());

            info!(
                "train time: {}, predict time: {}",
                results.train_time.last().unwrap(),
                results.predict_time.last().unwrap(),
            );

            info!(
                "train: {}, validation: {}",
                results.train_score.last().unwrap(),
                results.validation_score.last().unwrap(),
            )
        }

        results
    }
}
