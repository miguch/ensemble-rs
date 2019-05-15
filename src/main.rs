use rayon::prelude::*;

mod data_frame;
mod tree;
use data_frame::DataFrame;

use std::path::*;
use std::time;

static DATA_DIR: &str = "../data";

fn main() {
    let data_path = Path::new(DATA_DIR);
    let start = time::SystemTime::now();
    println!("Loading Train Data");
    let train_data = data_frame::read_csvs(
        (1..6)
            .into_iter()
            .map(|index| -> PathBuf { data_path.join(format!("train{}.csv", index)) })
            .collect(),
    );
    println!("Loading Label Data");
    let label_data = data_frame::read_csvs(
        (1..6)
            .into_iter()
            .map(|index| -> PathBuf { data_path.join(format!("label{}.csv", index)) })
            .collect(),
    );
    println!("Loading Test Data");
    let test_data = data_frame::read_csvs(
        (1..7)
            .into_iter()
            .map(|index| -> PathBuf { data_path.join(format!("test{}.csv", index)) })
            .collect(),
    );

    println!("Train data shape: {:?}", train_data.shape);
    println!("Label data shape: {:?}", label_data.shape);
    println!("Test data shape: {:?}", test_data.shape);
    println!("Load time: {}ms", start.elapsed().unwrap().as_millis());

    println!("{:?}", train_data.get_columns(vec![2, 3]));
}
