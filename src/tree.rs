use rayon::prelude::*;

enum TreeNode<V> {
    Leaf(V),
    Stem {
        /// The index of the feature of the split
        feature: usize,
        /// The value of the split
        param: V,
        /// The left child
        left: Box<TreeNode<V>>,
        /// The right child
        right: Box<TreeNode<V>>,
    },
}

unsafe impl<V: Send + Sync> Sync for TreeNode<V> {}
unsafe impl<V: Send + Sync> Send for TreeNode<V> {}

impl<V: Clone + Copy + Send + Sync> TreeNode<V> {}
