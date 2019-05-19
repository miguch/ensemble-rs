pub mod data_frame;
pub mod learner;
pub mod tree;
pub mod utils;

use rayon::prelude::*;
use std::collections::HashSet;
use utils::cross_validate;
use utils::numeric;

use pretty_env_logger;
#[macro_use]
extern crate log;

use crate::tree::DecisionTreeConfig;
use std::path::*;
use std::time;

static DATA_DIR: &str = "../data";

fn main() {
    pretty_env_logger::init();

    let data_path = Path::new(DATA_DIR);
    let start = time::SystemTime::now();
    debug!("Loading Train Data");
    let train_data = data_frame::read_csvs(
        (1..6)
            .into_par_iter()
            .map(|index| -> PathBuf { data_path.join(format!("train{}.csv", index)) })
            .collect(),
    );
    info!("Train data shape: {:?}", train_data.shape());
    debug!("Loading Label Data");
    let label_data = data_frame::read_csvs(
        (1..6)
            .into_par_iter()
            .map(|index| -> PathBuf { data_path.join(format!("label{}.csv", index)) })
            .collect(),
    );
    let samples_count = label_data.rows();
    let label_data = label_data.into_shape([1 as usize, samples_count]).unwrap();
    info!("Label data shape: {:?}", label_data.shape());
    //    debug!("Loading Test Data");
    //    let test_data = data_frame::read_csvs(
    //        (1..7)
    //            .into_iter()
    //            .map(|index| -> PathBuf { data_path.join(format!("test{}.csv", index)) })
    //            .collect(),
    //    );
    //    info!("Test data shape: {:?}", test_data.shape());

    info!("Load time: {}ms", start.elapsed().unwrap().as_millis());

    let tree_config = {
        let mut configs = HashSet::new();
        configs.insert(DecisionTreeConfig::MinSamplesLeaf(samples_count / 10000000));
        configs.insert(DecisionTreeConfig::MinSamplesSplit(
            samples_count / 10000000,
        ));
        configs.insert(DecisionTreeConfig::MaxBin(200));
        configs.insert(DecisionTreeConfig::MaxDepth(8));
        configs
    };

    let tree = tree::DecisionTree::new_with_config(tree_config);

    let folds = cross_validate::KFold::new(3);
    let results = folds.cross_validate(tree, &train_data, &label_data, numeric::r2_score);
    println!("{:?}", results);
    //    tree.fit(&train_data, &label_data);
}
