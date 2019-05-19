use ensembles_rs::data_frame;
use ensembles_rs::learner::Learner;
use ensembles_rs::tree;
use ensembles_rs::tree::DecisionTreeConfig;
use ensembles_rs::utils::numeric;
use ensembles_rs::utils::set_num_threads;
use rayon::prelude::*;
use std::collections::HashSet;
use num_cpus;

use log::*;

use pretty_env_logger;

use std::path::*;
use std::time;
use std::env;

static DATA_DIR: &str = "../data";

fn main() {
    pretty_env_logger::init();

    // Get threads number from cli params
    let params: Vec<String> = env::args().collect();
    let first_params = params.get(1);
    let threads = match first_params {
        Some(content) => content.parse::<usize>().unwrap(),
        None => num_cpus::get()
    };
    // Test different threads performance
    println!("Using {} threads", threads);
    set_num_threads(threads);

    let data_path = Path::new(DATA_DIR);
    let start = time::SystemTime::now();
    debug!("Loading Train Data");
    let train_data = data_frame::read_csvs(
        (1..2)
            .into_par_iter()
            .map(|index| -> PathBuf { data_path.join(format!("train{}.csv", index)) })
            .collect(),
    );
    info!("Train data shape: {:?}", train_data.shape());
    debug!("Loading Label Data");
    let label_data = data_frame::read_csvs(
        (1..2)
            .into_par_iter()
            .map(|index| -> PathBuf { data_path.join(format!("label{}.csv", index)) })
            .collect(),
    );
    let samples_count = label_data.rows();
    let label_data = label_data.into_shape([1 as usize, samples_count]).unwrap();
    info!("Label data shape: {:?}", label_data.shape());

    println!("Load time: {}ms", start.elapsed().unwrap().as_millis());

    let tree_config = {
        let mut configs = HashSet::new();
        configs.insert(DecisionTreeConfig::MinSamplesLeaf(samples_count / 10000000));
        configs.insert(DecisionTreeConfig::MinSamplesSplit(
            samples_count / 10000000,
        ));
        configs.insert(DecisionTreeConfig::MaxBin(100));
        configs.insert(DecisionTreeConfig::MaxDepth(3));
        configs
    };

    let tree = tree::DecisionTree::new_with_config(tree_config);


    {
        let mut model = tree.clone();
        let start = time::SystemTime::now();
        model.fit(&train_data, &label_data);
        println!("{} threads training time: {}ms", threads, start.elapsed().unwrap().as_millis());
        let start = time::SystemTime::now();
        let result = model.predict(&train_data);
        println!("Train score: {}", numeric::r2_score(&label_data, &result));
        println!("{} threads predict time: {}ms", threads, start.elapsed().unwrap().as_millis());
    }


}
