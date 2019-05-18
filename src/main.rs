mod data_frame;
mod learner;
mod tree;
mod utils;
use learner::*;
use rayon::prelude::*;
use std::collections::HashSet;
use utils::sort_array::*;

#[macro_use]
extern crate ndarray;

use crate::tree::DecisionTreeConfig;
use ndarray::prelude::*;
use ndarray::*;
use std::path::*;
use std::time;

static DATA_DIR: &str = "../data";

fn main() {
    let data_path = Path::new(DATA_DIR);
    let start = time::SystemTime::now();
    println!("Loading Train Data");
    let train_data = data_frame::read_csvs(
        (1..2)
            .into_par_iter()
            .map(|index| -> PathBuf { data_path.join(format!("train{}.csv", index)) })
            .collect(),
    );
    println!("Train data shape: {:?}", train_data.shape());
    println!("Loading Label Data");
    let label_data = data_frame::read_csvs(
        (1..2)
            .into_par_iter()
            .map(|index| -> PathBuf { data_path.join(format!("label{}.csv", index)) })
            .collect(),
    );
    let samples_count = label_data.rows();
    let label_data = label_data.into_shape([1 as usize, samples_count]).unwrap();
    println!("Label data shape: {:?}", label_data.shape());
    //    println!("Loading Test Data");
    //    let test_data = data_frame::read_csvs(
    //        (1..7)
    //            .into_iter()
    //            .map(|index| -> PathBuf { data_path.join(format!("test{}.csv", index)) })
    //            .collect(),
    //    );
    //    println!("Test data shape: {:?}", test_data.shape());

    println!("Load time: {}ms", start.elapsed().unwrap().as_millis());

    let tree_config = {
        let mut configs = HashSet::new();
        configs.insert(DecisionTreeConfig::MinSamplesLeaf(samples_count / 10000000));
        configs.insert(DecisionTreeConfig::MinSamplesSplit(
            samples_count / 10000000,
        ));
        configs.insert(DecisionTreeConfig::MaxBin(200));
        configs
    };

    let mut tree = tree::DecisionTree::new_with_config(tree_config);

    tree.fit(&train_data, &label_data);
}
