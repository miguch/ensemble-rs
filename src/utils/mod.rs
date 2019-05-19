pub mod cross_validate;
pub mod numeric;
pub mod sort_array;

use rayon::ThreadPoolBuilder;

pub fn set_num_threads(workers: usize) {
    let tpb = ThreadPoolBuilder::new();
    let tpb = tpb.num_threads(workers);
    tpb.build_global().unwrap();
}
