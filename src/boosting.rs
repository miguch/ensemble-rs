use crate::data_frame::*;
use crate::learner::*;
use crate::utils::numeric;
use indicatif;
use log::*;
use rand::seq::SliceRandom;
use rayon::prelude::*;
use std::cmp::Ordering;

#[derive(Clone)]
pub struct GradientBoosting<L> {
    pub weak_learner: L,
    /// Fitted learners, each learner is initially cloned from weak_learner
    pub learners: Vec<L>,
    /// How each learner affect the model
    learning_rates: Vec<f64>,
    /// The number of weak learners to fit
    pub max_iterations: usize,
    /// The fraction of samples to train fir individual weak learner
    pub sub_sample: f64,
    /// The initial value of the model
    init_value: V
}

#[derive(PartialEq, Clone)]
pub enum GBDTConfig {
    MaxIterations(usize),
    SubSample(f64),
}

impl<L: Learner + Clone + Sync + Send> GradientBoosting<L> {
    pub fn new(learner: L) -> Self {
        Self {
            weak_learner: learner,
            learners: Vec::with_capacity(100),
            learning_rates: vec![],
            max_iterations: 100,
            sub_sample: 1.0,
            init_value: 0.0
        }
    }

    // Use vec here because floating points can't be hash
    pub fn with_config(configs: Vec<GBDTConfig>, learner: L) -> Self {
        let mut boost = Self::new(learner);
        for item in configs {
            use crate::boosting::GBDTConfig::*;
            match item {
                MaxIterations(i) => {
                    boost.max_iterations = i;
                    boost.learners.reserve(i);
                }
                SubSample(s) => {
                    boost.sub_sample = s;
                }
            }
        }
        boost
    }

    // Returns the weak learner trained at this step
    pub fn train_one_step(&mut self, x: &DataFrame, residuals: &DataFrame) -> L {
        let mut learner = self.weak_learner.clone();

        learner.fit(x, residuals);

        learner
    }

    /// Returns the sub sample x, residual and feature orders
    fn choose_subsample(&self, x: &DataFrame, residual: &DataFrame) -> (DataFrame, DataFrame) {
        let mut rng = rand::thread_rng();
        let mut orders: Vec<usize> = (0..x.rows()).collect();
        orders.shuffle(&mut rng);
        let sub_sample_size = (self.sub_sample * orders.len() as f64) as usize;
        orders.resize(sub_sample_size, 0);

        let mut buffer = Vec::with_capacity(sub_sample_size);
        let mut residual_buffer = Vec::with_capacity(sub_sample_size);
        for index in orders {
            buffer.extend(x.row(index).iter());
            residual_buffer.push(residual[[0, index]]);
        }
        (
            DataFrame::from_shape_vec((sub_sample_size, x.cols()), buffer).unwrap(),
            DataFrame::from_shape_vec((1, sub_sample_size), residual_buffer).unwrap(),
        )
    }
}

impl<L: Learner + Clone + Sync + Send> Learner for GradientBoosting<L> {
    fn fit(&mut self, x: &DataFrame, y: &DataFrame) {
        if !self.learners.is_empty() {
            panic!("Model is already trained!");
        }

        self.init_value = y.row(0).iter().sum::<V>() / y.cols() as V;

        let samples = y.cols();
        // Initialize F_0(x) to constant 0
        let mut model_pred = DataFrame::from_shape_vec((1, y.cols()), vec![self.init_value; samples]).unwrap();

        info!("Start training...");

        //Add a progress bar here
        let indif = indicatif::ProgressBar::new(self.max_iterations as u64);
        for _i in 0..self.max_iterations {
            let residual = y - &model_pred;
            let (sub_x, sub_residual) = self.choose_subsample(x, &residual);
            let model = self.train_one_step(&sub_x, &sub_residual);

            let new_pred = model.predict(&x);

            // Parallel line search to fine the best lr
            let (best_lr, _r2) = (0..101).into_par_iter().map(|i| {
                let lr = 0.01 * i as f64;
                let pred = &model_pred + &(&new_pred * lr);
                let r2 =  numeric::r2_score(y, &pred);
                (lr, r2)
            }).max_by(|a, b| {
                if a.1 < b.1 {
                    Ordering::Less
                } else if a.1 > b.1 {
                    Ordering::Greater
                } else {
                    Ordering::Equal
                }
            }).unwrap();
            info!("lr {} at step {}.", best_lr, _i);

            self.learning_rates.push(best_lr);
            model_pred = model_pred + new_pred * best_lr;

            info!("Pred Score: {}\n", numeric::r2_score(y, &model_pred));
            println!("{}", numeric::r2_score(y, &model_pred));

            self.learners.push(model);
            // Update progress bar
            indif.inc(1);
        }
    }

    fn predict(&self, df: &DataFrame) -> DataFrame {
        if self.learners.is_empty() {
            panic!("Model is not trained!");
        }
        let mut model_pred =
            DataFrame::from_shape_vec((1, df.rows()), vec![self.init_value; df.rows()]).unwrap();
        for i in 0..self.learners.len() {
            let curr_pred = self.learners[i].predict(df);
            model_pred = model_pred + curr_pred * self.learning_rates[i];
        }

        model_pred
    }
}
