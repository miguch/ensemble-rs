use crate::data_frame::*;
/// Adapted from ndarray/examples
use ndarray::prelude::*;
use ndarray::{Data};

use rayon::prelude::*;

use std::cmp::Ordering;

// Type invariant: Each index appears exactly once
#[derive(Clone, Debug)]
pub struct Permutation {
    pub indices: Vec<usize>,
}

impl Permutation {
    /// Checks if the permutation is correct
    #[allow(dead_code)]
    pub fn from_indices(v: Vec<usize>) -> Result<Self, ()> {
        let perm = Permutation { indices: v };
        if perm.correct() {
            Ok(perm)
        } else {
            Err(())
        }
    }

    #[allow(dead_code)]
    fn correct(&self) -> bool {
        let axis_len = self.indices.len();
        let mut seen = vec![false; axis_len];
        for &i in &self.indices {
            match seen.get_mut(i) {
                None => return false,
                Some(s) => {
                    if *s {
                        return false;
                    } else {
                        *s = true;
                    }
                }
            }
        }
        true
    }
}

pub trait SortArray {
    /// ***Panics*** if `axis` is out of bounds.
    fn identity(&self, axis: Axis) -> Permutation;
    fn sort_column(&self, col_index: usize) -> Permutation;
}

pub trait PermuteArray {
    type Elem;
    fn permute(self, perm: &Permutation) -> Array2<Self::Elem>
    where
        Self::Elem: Clone;
    fn permute_with_index(self, perm: &Permutation) -> Array2<Self::Elem>
    where
        Self::Elem: Clone;
}

impl<A, S> SortArray for ArrayBase<S, Dim<[usize; 2]>>
where
    A: PartialOrd + Sync,
    S: Data<Elem = A> + Sync,
{
    fn identity(&self, axis: Axis) -> Permutation {
        Permutation {
            indices: (0..self.len_of(axis)).collect(),
        }
    }

    fn sort_column(&self, col_index: usize) -> Permutation {
        let mut perm = self.identity(Axis(0));
        perm.indices.par_sort_by(|&a_label, &b_label| {
            let a = &self[[a_label, col_index]];
            let b = &self[[b_label, col_index]];
            a.partial_cmp(&b).unwrap_or(Ordering::Equal)
        });
        perm
    }
}

impl PermuteArray for Array2<V> {
    type Elem = V;

    fn permute(self, perm: &Permutation) -> DataFrame {
        let axis_len = self.len_of(Axis(0));

        assert_eq!(axis_len, perm.indices.len());
        debug_assert!(perm.correct());

        let mut v: Vec<V> = Vec::with_capacity(self.len());

        for index in &perm.indices {
            v.extend(self.row(*index).iter().cloned());
        }
        
        Array2::from_shape_vec((self.rows(), self.cols()), v).unwrap()
    }

    fn permute_with_index(self, perm: &Permutation) -> DataFrame {
        let axis_len = self.len_of(Axis(0));

        assert_eq!(axis_len, perm.indices.len());
        debug_assert!(perm.correct());

        let mut v: Vec<V> = Vec::with_capacity(self.len());

        for index in &perm.indices {
            v.extend(self.row(*index).iter().cloned());
            v.push(*index as V)
        }
        
        Array2::from_shape_vec((self.rows(), self.cols() + 1), v).unwrap()
    }
}
