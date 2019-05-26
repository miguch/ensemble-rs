use ensembles_rs::boosting::*;
use ensembles_rs::data_frame;
use ensembles_rs::tree;
use ensembles_rs::tree::DecisionTreeConfig;
use ensembles_rs::utils::cross_validate;
use ensembles_rs::utils::numeric;
use rayon::prelude::*;
use std::collections::HashSet;

use log::*;

use pretty_env_logger;

use std::path::*;
use std::time;

static DATA_DIR: &str = "../data";

fn main() {
    pretty_env_logger::try_init_timed_custom_env("ENSEM_LOG").unwrap();

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

    info!("Load time: {}ms", start.elapsed().unwrap().as_millis());

    let tree_config = {
        let mut configs = HashSet::new();
        configs.insert(DecisionTreeConfig::MinSamplesLeaf(samples_count / 10000000));
        configs.insert(DecisionTreeConfig::MinSamplesSplit(
            samples_count / 10000000,
        ));
        configs.insert(DecisionTreeConfig::MaxBin(300));
        configs.insert(DecisionTreeConfig::MaxDepth(2));
        configs
    };

    let tree = tree::DecisionTree::new_with_config(tree_config);

    let boost_config = vec![GBDTConfig::MaxIterations(150), GBDTConfig::SubSample(0.1)];

    let boost = GradientBoosting::with_config(boost_config, tree);

    let folds = cross_validate::KFold::new(3);
    let results = folds.cross_validate(boost, &train_data, &label_data, numeric::r2_score);
    println!("{:?}", results);
}
