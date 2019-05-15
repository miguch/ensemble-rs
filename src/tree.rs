use rayon::prelude::*;

enum TreeNode<P, V> {
    Leaf(V),
    Stem {
        param: P,
        left: Box<TreeNode<P, V>>,
        right: Box<TreeNode<P, V>>,
    },
}
