use csv;

use rayon::prelude::*;
use std::fs::File;
use std::path::PathBuf;

use ndarray::*;
use std::io::{BufWriter, Write};

/// The type used for values in dataframe
pub type V = f64;

pub type DataFrame = Array2<V>;

#[derive(Debug, Clone)]
pub struct StatsFeature {
    pub cols_mean: Array1<V>,
    pub cols_std: Array1<V>,
}

impl StatsFeature {
    #[allow(dead_code)]
    pub fn from_df_cols(df: &DataFrame) -> Self {
        Self {
            cols_mean: df.mean_axis(Axis(0)),
            cols_std: df.std_axis(Axis(0), 1.0),
        }
    }

    #[allow(dead_code)]
    pub fn from_df_rows(df: &DataFrame) -> Self {
        Self {
            cols_mean: df.mean_axis(Axis(1)),
            cols_std: df.std_axis(Axis(1), 1.0),
        }
    }
}

pub fn df_from_data(buf: Vec<Vec<V>>) -> DataFrame {
    if buf.is_empty() {
        Array2::zeros((0, 0))
    } else {
        let shape = (buf.len(), buf[0].len());
        let flatten = buf.into_iter().flatten().collect();
        let arr = Array2::from_shape_vec(shape, flatten).unwrap();
        arr
    }
}

#[allow(dead_code)]
pub fn sort_f64_vec(v: &mut Vec<V>) {
    use std::cmp::Ordering;
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
}

pub fn read_csvs(file_paths: Vec<PathBuf>) -> DataFrame {
    let mut data_frame = Vec::new();
    let frames: Vec<Vec<Vec<V>>> = file_paths.par_iter().map(|name| load_csv(name)).collect();

    for frame in frames {
        data_frame.extend(frame);
    }

    df_from_data(data_frame)
}

pub fn save_csv(df: &DataFrame, file_path: PathBuf, headers: &[&str]) {
    assert_eq!(headers.len(), df.cols());
    let file = File::create(file_path).unwrap();
    let mut writer = BufWriter::new(file);
    writer
        .write(headers.join(",").as_bytes())
        .expect("Unable to write data");
    writer.write(b"\n").expect("Unable to write data");
    for i in 0..df.rows() {
        let row: Vec<String> = (0..df.cols())
            .into_iter()
            .map(|c| df[[i, c]].to_string())
            .collect();
        writer
            .write(row.join(",").as_bytes())
            .expect("Unable to write data");
        writer.write(b"\n").expect("Unable to write data");
    }
}

fn load_csv(file_path: &PathBuf) -> Vec<Vec<V>> {
    let file = match File::open(file_path) {
        Err(e) => panic!("couldn't open {}: {}", file_path.display(), e),
        Ok(f) => f,
    };
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_reader(file);
    let mut data_frame = Vec::new();
    for result in reader.records() {
        let record = result.unwrap();
        data_frame.push(
            record
                .iter()
                .map(|e: &str| e.parse::<V>().unwrap())
                .collect(),
        );
    }
    data_frame
}
