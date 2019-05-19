use crate::data_frame::*;

pub trait Learner {
    /// df: `[samples, features]` y: `[1, samples]`
    fn fit(&mut self, x: &DataFrame, y: &DataFrame);

    /// df: `[sample, features]`
    fn predict(&self, df: &DataFrame) -> DataFrame;
}
