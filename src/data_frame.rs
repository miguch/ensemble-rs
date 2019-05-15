use csv;
use rayon::prelude::*;
use std::error::Error;
use std::fs::File;
use std::io::prelude::*;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug)]
pub struct DataFrame<V> {
    pub data: Vec<Vec<V>>,
    pub shape: (usize, usize),
}

impl<V: Copy + Clone> DataFrame<V> {
    pub fn new(buf: Vec<Vec<V>>) -> Self {
        if buf.is_empty() {
            Self {
                data: buf,
                shape: (0, 0),
            }
        } else {
            let res = Self {
                shape: (buf.len(), buf[0].len()),
                data: buf,
            };
            for row in &res.data {
                if row.len() != res.shape.1 {
                    panic!("Row size inconsistent");
                }
            }
            res
        }
    }

    pub fn get_rows(&self, indexes: Vec<usize>) -> Self {
        let mut result = Vec::new();
        for index in indexes {
            result.push(self.data[index].clone());
        }
        DataFrame::new(result)
    }

    pub fn get_columns(&self, indexes: Vec<usize>) -> Self {
        let mut result = Vec::new();
        for row in &self.data {
            let mut row = Vec::new();
            row.reserve(indexes.len());
            for index in &indexes {
                row.push(row[*index]);
            }
            result.push(row)
        }
        DataFrame::new(result)
    }

}

pub fn read_csvs(file_paths: Vec<PathBuf>) -> DataFrame<f64> {
    let mut data_frame = Vec::new();
    let frames: Vec<Vec<Vec<f64>>> = file_paths.par_iter().map(|name| load_csv(name)).collect();

    for frame in frames {
        data_frame.extend(frame);
    }

    DataFrame::<f64>::new(data_frame)
}

fn load_csv(file_path: &PathBuf) -> Vec<Vec<f64>> {
    let mut file = match File::open(file_path) {
        Err(e) => panic!("couldn't open {}: {}", file_path.display(), e),
        Ok(mut f) => f,
    };
    let mut data_frame = Vec::new();
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_reader(file);
    for result in reader.records() {
        let record = result.unwrap();
        data_frame.push(
            record
                .iter()
                .map(|e: &str| e.parse::<f64>().unwrap())
                .collect(),
        );
    }
    data_frame
}
