use ensembles_rs::data_frame;
use ensembles_rs::learner::Learner;
use ensembles_rs::random_forest::*;
use ensembles_rs::tree;
use ensembles_rs::tree::DecisionTreeConfig;
use ensembles_rs::utils::numeric;
use rayon::prelude::*;
use std::collections::HashSet;
use serde_json;

use log::*;

use pretty_env_logger;

use std::path::*;
use std::time;
use std::fs::File;
use std::io::{Write, Read};

static DATA_DIR: &str = "../data";
static MODEL_DIR: &str = "../model";

fn main() {
    pretty_env_logger::init();

    let data_path = Path::new(DATA_DIR);
    let model_path = Path::new(MODEL_DIR);
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
    debug!("Loading Test Data");
    let test_data = data_frame::read_csvs(
        (1..7)
            .map(|index| -> PathBuf { data_path.join(format!("test{}.csv", index)) })
            .collect(),
    );
    info!("Test data shape: {:?}", test_data.shape());

    info!("Load time: {}ms", start.elapsed().unwrap().as_millis());

    let tree_config = {
        let mut configs = HashSet::new();
        configs.insert(DecisionTreeConfig::MinSamplesLeaf(samples_count / 2000000));
        configs.insert(DecisionTreeConfig::MinSamplesSplit(samples_count / 2000000));
        configs.insert(DecisionTreeConfig::MaxBin(500));
        configs.insert(DecisionTreeConfig::MaxFeatures(4));
        configs
    };

    let tree = tree::DecisionTree::new_with_config(tree_config);

    let forest_config = vec![
        RandomForestConfig::SubSample(0.15),
        RandomForestConfig::NEstimators(100),
    ];

    let mut rf = RandomForest::from_configs(tree, forest_config);

    if model_path.join("RF.json").exists() {
        println!("Previous model found, reading in model...");
        let mut serial: String = String::new();
        File::open(model_path.join("RF.json")).unwrap().read_to_string(&mut serial).unwrap();
        rf = serde_json::from_str(&serial).unwrap();
    }

    rf.fit(&train_data, &label_data);

    let result = rf.predict(&train_data);
    println!("Train score: {}", numeric::r2_score(&label_data, &result));

    let result = rf.predict(&test_data);
    let mut csv_data = vec![];
    // to csv data frame
    for i in 0..result.cols() {
        csv_data.push((i + 1) as data_frame::V);
        csv_data.push(result[[0, i]]);
    }
    let csv_data = data_frame::DataFrame::from_shape_vec((result.cols(), 2), csv_data).unwrap();
    println!("Writing to file");
    data_frame::save_csv(&csv_data, data_path.join("RF.csv"), &["id", "Predicted"]);

    let serial = serde_json::to_string(&rf).unwrap();
    let mut file = File::create(model_path.join("RF.json")).unwrap();
    file.write_all(serial.as_bytes()).unwrap();
}
