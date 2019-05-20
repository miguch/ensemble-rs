use crate::data_frame::*;
use ndarray::*;

/// a and b should both be of size (1, sample_len)
pub fn mse_score(a: &DataFrame, b: &DataFrame) -> V {
    let mut result: V = 0.0;
    assert_eq!(a.shape(), b.shape());
    for i in 0..a.cols() {
        result += (a[[0, i]] - b[[0, i]]).powi(2);
    }
    (result / a.cols() as V)
}

/// a and b should both be of size (1, sample_len)
pub fn slice_mse_score(a: &[V], b: &[V]) -> V {
    let mut result: V = 0.0;
    assert_eq!(a.len(), b.len());
    for i in 0..a.len() {
        result += (a[i] - b[i]).powi(2);
    }
    (result / a.len() as V)
}

/// Calculate the square sum of all elements in a minus the mean of a
/// size of a: (1, sample_len)
pub fn col_variance(a: &DataFrame) -> V {
    let mean = a.mean_axis(Axis(1))[0];
    let mut result: V = 0.0;
    for i in 0..a.cols() {
        result += (a[[0, i]] - mean).powi(2);
    }
    result
}

/// Calculate the square sum of all elements in a minus the mean of a
pub fn slice_variance(a: &[V]) -> V {
    let mean: V = a.iter().sum::<V>() / a.len() as V;
    let mut result: V = 0.0;
    for i in a {
        result += (*i - mean).powi(2);
    }
    result
}

/// a and b should both be of size (1, sample_len)
pub fn r2_score(true_y: &DataFrame, pred_y: &DataFrame) -> V {
    let true_mean = true_y.mean_axis(Axis(1))[0];
    assert_eq!(true_y.shape(), pred_y.shape());
    let mut square_error: V = 0.0;
    let mut mean_diff: V = 0.0;
    for i in 0..true_y.cols() {
        square_error += (true_y[[0, i]] - pred_y[[0, i]]).powi(2);
        mean_diff += (true_y[[0, i]] - true_mean).powi(2);
    }
    1.0 - (square_error / mean_diff)
}

// Test will only pass if V is f64
#[cfg(test)]
mod test {
    use crate::utils::numeric::{mse_score, r2_score};
    use ndarray::*;

    #[test]
    fn mse_test() {
        let a = array![[3.1, -0.5, 2.66, 7.6]];
        let b = array![[2.5, 0.0, 2.4, 8.4]];
        assert_eq!(0.32940000000000036, mse_score(&a, &b));
    }

    #[test]
    fn r2_test() {
        let a = array![[3.1, -0.5, 2.66, 7.6]];
        let b = array![[2.5, 0.0, 2.4, 9.45]];
        assert_eq!(0.8770610511923288, r2_score(&a, &b));
    }
}
