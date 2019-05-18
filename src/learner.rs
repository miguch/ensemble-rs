use crate::data_frame::*;

pub trait Learner {
    /// df: `[{features}, label]`
    fn fit(&mut self, x: &DataFrame, y: &DataFrame);

    ///df: `[{features}]`
    fn predict(&self, df: &DataFrame);
}
