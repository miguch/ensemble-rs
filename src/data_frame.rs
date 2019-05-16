use csv;
use rayon::prelude::*;
use std::error::Error;
use std::fs::File;
use std::path::PathBuf;

use ndarray::{arr2, Array2, Shape};

#[derive(Clone, Debug)]
pub struct DataFrame<V> {
    pub data: Vec<V>,
    pub shape: (usize, usize),
}

impl<V: Copy + Clone + Send + Sync + Default> DataFrame<V> {
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            shape: (0, 0),
        }
    }

    pub fn from_data(buf: Vec<Vec<V>>) -> Self {
        if buf.is_empty() {
            Self {
                data: Vec::new(),
                shape: (0, 0),
            }
        } else {
            let shape = (buf.len(), buf[0].len());
            for row in &buf {
                if row.len() != shape.1 {
                    panic!("Row size inconsistent");
                }
            }
            let mut data = Vec::with_capacity(shape.1 * shape.0);
            for row in buf {
                data.extend(row);
            }
            Self { data, shape }
        }
    }

    pub fn get_rows(&self, indexes: &[usize]) -> Self {
        let mut result = Vec::with_capacity(indexes.len() * self.shape.1);
        for index in indexes {
            result.extend(&self.data[(index * self.shape.1)..((index + 1) * self.shape.1)]);
        }
        Self {
            data: result,
            shape: (indexes.len(), self.shape.1),
        }
    }

    pub fn get_columns(&self, indexes: &[usize]) -> Self {
        let mut result = Vec::with_capacity(indexes.len() * self.shape.0);
        // Get column data in each row
        for i in 0..self.shape.0 {
            let index_base = i * self.shape.1;
            result.extend::<Vec<V>>(
                indexes
                    .iter()
                    .map(|index| -> V { self.data[index_base + index] })
                    .collect(),
            )
        }
        Self {
            data: result,
            shape: (self.shape.0, indexes.len()),
        }
    }

    pub fn reshape(&mut self, new_shape: (usize, usize)) {
        if new_shape.1 * new_shape.0 != self.shape.1 * self.shape.0 {
            panic!(
                "Unable to resize. old size: {:?}, new size: {:?}",
                self.shape, new_shape
            );
        }
        self.shape = new_shape;
    }
}

use std::fmt;
impl<V: fmt::Display> fmt::Display for DataFrame<V> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "[\n")?;
        for i in 0..self.shape.0 {
            write!(f, " [")?;
            let base = i * self.shape.1;
            for k in 0..self.shape.1 {
                write!(f, "{}", self.data[base + k])?;
                if k != self.shape.1 - 1 {
                    write!(f, ",")?;
                }
            }
            write!(f, "],\n")?;
        }
        write!(f, "]")
    }
}

pub fn read_csvs(file_paths: Vec<PathBuf>) -> DataFrame<f64> {
    let mut data_frame = Vec::new();
    let frames: Vec<Vec<Vec<f64>>> = file_paths.par_iter().map(|name| load_csv(name)).collect();

    for frame in frames {
        data_frame.extend(frame);
    }

    DataFrame::<f64>::from_data(data_frame)
}

fn load_csv(file_path: &PathBuf) -> Vec<Vec<f64>> {
    let mut file = match File::open(file_path) {
        Err(e) => panic!("couldn't open {}: {}", file_path.display(), e),
        Ok(mut f) => f,
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
                .map(|e: &str| e.parse::<f64>().unwrap())
                .collect(),
        );
    }
    data_frame
}
