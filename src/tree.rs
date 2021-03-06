use crate::data_frame;
use crate::learner::*;
use crate::tree::NodeInfo::Stem;
use crate::utils::numeric::*;
use crate::utils::sort_array::*;
use data_frame::*;
use log::*;
use num_traits::*;
use rand::prelude::*;
use serde::{Serialize, Deserialize};

use rayon::prelude::*;
use std::cmp::Ordering;

use std::collections::*;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionTree {
    /// Nodes in the tree as a list of nodes
    pub nodes: Vec<TreeNode<V>>,
    /// Maximum depth the tree can grow
    pub max_depth: usize,
    /// Maximum number of features to look for a split
    pub max_features: usize,
    /// The minimum number of samples to perform a split
    pub min_samples_split: usize,
    /// The maximum number of samples for a node to be a leaf
    pub min_samples_leaf: usize,
    /// The number of bins feature values will be bucketed into
    pub max_bin: usize,
}

#[derive(PartialEq, Eq, Hash, Clone, Serialize, Deserialize)]
pub enum DecisionTreeConfig {
    MaxDepth(usize),
    MaxFeatures(usize),
    MinSamplesSplit(usize),
    MinSamplesLeaf(usize),
    MaxBin(usize),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NodeInfo<V> {
    Leaf,
    Stem {
        /// The index of the feature of the split
        feature: usize,
        /// The value of the split
        param: V,
        /// The left child
        left: usize,
        /// The right child
        right: usize,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TreeNode<V> {
    pub value: V,
    pub index: usize,
    pub depth: usize,
    pub variance: V,
    pub info: NodeInfo<V>,
}

/// Used to create thread-safe parallel split by node
struct SplitInfo<V> {
    /// The index of the node to be split
    pub index: usize,
    /// The new stem the split node about to become
    pub new_stem: NodeInfo<V>,
    /// The new variance of the current node
    pub new_variance: V,
    /// The left node object
    pub left_node: TreeNode<V>,
    /// The right node object
    pub right_node: TreeNode<V>,
    /// The indexes of samples to be in the left node
    pub left_indexes: HashSet<usize>,
    /// The indexes of samples to be in the right node
    pub right_indexes: HashSet<usize>,
}

// Parallel stuff
unsafe impl<V: Send + Sync> Sync for NodeInfo<V> {}
unsafe impl<V: Send + Sync> Send for NodeInfo<V> {}
unsafe impl<V: Send + Sync> Sync for TreeNode<V> {}
unsafe impl<V: Send + Sync> Send for TreeNode<V> {}
unsafe impl Sync for DecisionTree {}
unsafe impl Send for DecisionTree {}
unsafe impl<V: Send + Sync> Sync for SplitInfo<V> {}
unsafe impl<V: Send + Sync> Send for SplitInfo<V> {}

impl DecisionTree {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            max_depth: std::usize::MAX,
            max_features: std::usize::MAX,
            min_samples_leaf: 1,
            min_samples_split: 3,
            max_bin: 255,
        }
    }

    pub fn new_with_config(config: HashSet<DecisionTreeConfig>) -> Self {
        let mut tree = Self::new();
        for item in config {
            match item {
                DecisionTreeConfig::MaxDepth(d) => {
                    tree.max_depth = d;
                    tree.nodes.reserve(pow(2, d + 1) - 1);
                }
                DecisionTreeConfig::MaxFeatures(f) => tree.max_features = f,
                DecisionTreeConfig::MinSamplesSplit(s) => tree.min_samples_split = s,
                DecisionTreeConfig::MinSamplesLeaf(s) => tree.min_samples_leaf = s,
                DecisionTreeConfig::MaxBin(bin) => tree.max_bin = bin,
            }
        }
        tree
    }

    fn build_model(&mut self, feature_order: Vec<Vec<usize>>, df: &DataFrame, labels: &DataFrame) {
        let stats_info = StatsFeature::from_df_cols(&df);
        let root = TreeNode {
            value: stats_info.cols_mean[stats_info.cols_mean.len() - 1],
            index: 0,
            depth: 0,
            variance: col_variance(labels),
            info: NodeInfo::Leaf,
        };

        self.nodes.push(root);

        //records at each tree node by index, data will be dropped after use to save memory
        let mut node_sample_index = HashMap::<usize, HashSet<usize>>::new();
        //Root node contains all records
        node_sample_index.insert(0, (0..df.rows()).collect());

        //contains indexes to nodes to be split
        let mut current_nodes: Vec<usize> = vec![0];

        // Loop till the tree has no nodes to be further split
        // Each loop will grow another level in the resulting tree
        // a node stop being split when it reaches the max depth
        // or its number of records is lower than min_sample_split
        loop {
            if current_nodes.is_empty() {
                break;
            }

            // Parallel perform split for current leaf nodes
            // Get split info about the current node and two children
            let next_nodes_pairs: Vec<Option<SplitInfo<V>>> = current_nodes
                .par_iter()
                .map(|node_index| {
                    // Stop split when reach max depthor
                    if self.nodes[*node_index].depth >= self.max_depth {
                        return None;
                    }

                    let curr_samples = node_sample_index.get(node_index).unwrap();
                    // No split when there are not many samples in node or
                    // samples can not be divided to bins
                    if curr_samples.len() < self.min_samples_split
                    {
                        return None;
                    }

                    let curr_bins = self.max_bin.min(curr_samples.len());

                    // local thread Random generator
                    let mut rng = rand::thread_rng();

                    // split a node
                    let mut features: Vec<usize> = (0..df.cols()).collect();
                    features.shuffle(&mut rng);
                    // only take the max_features number of features from the list
                    features.resize(self.max_features.min(df.cols()), 0);

                    let bin_size = curr_samples.len() / curr_bins;
                    // split points of bins, k bins means k-1 split points
                    let bins: Vec<usize> = (1..curr_bins)
                        .into_iter()
                        .map(|b_i| bin_size * b_i)
                        .collect();

                    // parallel perform split for each feature
                    // get the indexes for the splitting value and the square errors of each resulting split
                    // (feature_index, feature_split_point, variance, left_variance, right_variance, split_index)
                    let feature_split: Vec<(usize, usize, V, V, V, usize)> = features
                        .par_iter()
                        .map(|feat_index| {
                            // split position
                            let mut best_split: usize = 0;
                            // sum of two children's square error after split
                            let mut best_sqr_err: V = self.nodes[*node_index].variance;
                            let mut best_right_err: V = 0.0;
                            let mut best_left_err: V = 0.0;
                            let mut feat_labels = Vec::with_capacity(curr_samples.len());
                            let orders: &Vec<usize> = &feature_order[*feat_index];
                            let mut bins_index = vec![std::usize::MAX; curr_bins - 1];
                            for index in orders {
                                if curr_samples.contains(index) {
                                    // set split index for k-1 split points
                                    if feat_labels.len() % bin_size == 0
                                        && feat_labels.len() != 0
                                        && feat_labels.len() < bin_size * curr_bins
                                    {
                                        bins_index[feat_labels.len() / bin_size - 1] = *index;
                                    }
                                    feat_labels.push(labels[[0, *index]]);
                                }
                            }
                            // search each bins for resulting split info
                            for bin in &bins {
                                let (left, right) = feat_labels.split_at(*bin);
                                // let left_weight = *bin as V / curr_samples.len() as V;
                                let left_sqr_err = slice_variance(left);
                                let right_sqr_err = slice_variance(right);
                                let curr_sqr_err = left_sqr_err + right_sqr_err;
                                if curr_sqr_err < best_sqr_err {
                                    best_split = *bin;
                                    best_sqr_err = curr_sqr_err;
                                    best_left_err = left_sqr_err;
                                    best_right_err = right_sqr_err;
                                }
                            }
                            (
                                *feat_index,
                                best_split,
                                best_sqr_err,
                                best_left_err,
                                best_right_err,
                                // don't panic when this if there is no split!
                                if best_split == 0 {
                                    1
                                } else {
                                    bins_index[best_split / bin_size - 1]
                                },
                            )
                        })
                        .collect();

                    // sort splits to find the split that minimize the new variance
                    let min_split = feature_split
                        .iter()
                        .min_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(Ordering::Equal));

                    match min_split {
                        None => None,
                        Some(split) => {
                            // Not enough to form a leaf
                            if split.1 < self.min_samples_leaf
                                || curr_samples.len() - split.1 < self.min_samples_leaf
                                || split.1 == 0
                                || curr_samples.len() - split.1 == 0
                            {
                                return None;
                            }

                            // samples in children after split
                            let mut left_index = HashSet::with_capacity(split.1);
                            let mut right_index =
                                HashSet::with_capacity(curr_samples.len() - split.1);

                            // find the split point value
                            let orders: &Vec<usize> = &feature_order[split.0];
                            let mut split_val: V = V::infinity();
                            let mut index_to_write = &mut left_index;
                            let mut right_label_sum: V = 0.0;
                            let mut left_label_sum: V = 0.0;
                            let mut sum_to_add = &mut left_label_sum;
                            for index in orders {
                                if curr_samples.contains(index) {
                                    index_to_write.insert(*index);
                                    *sum_to_add += labels[[0, *index]];
                                    if *index == split.5 {
                                        split_val = df[[*index, split.0]];
                                        // start recording right children's samples
                                        index_to_write = &mut right_index;
                                        sum_to_add = &mut right_label_sum;
                                    }
                                }
                            }

                            // create left and right node
                            let left_node = TreeNode {
                                value: left_label_sum / left_index.len() as V,
                                index: 0,
                                depth: self.nodes[*node_index].depth + 1,
                                variance: split.3,
                                info: NodeInfo::Leaf,
                            };
                            let right_node = TreeNode {
                                value: right_label_sum / right_index.len() as V,
                                index: 0,
                                depth: self.nodes[*node_index].depth + 1,
                                variance: split.4,
                                info: NodeInfo::Leaf,
                            };

                            Some(SplitInfo {
                                index: *node_index,
                                new_stem: Stem {
                                    feature: split.0,
                                    param: split_val,
                                    left: 0,
                                    right: 0,
                                },
                                new_variance: split.2,
                                left_node,
                                right_node,
                                left_indexes: left_index,
                                right_indexes: right_index,
                            })
                        }
                    }
                })
                .collect();

            // prepare for next round grow
            node_sample_index.clear();
            current_nodes.clear();

            // Update everything from split info in serial
            for node_pair in next_nodes_pairs {
                if let None = node_pair {
                    continue;
                } else if let Some(mut split_info) = node_pair {
                    let curr_nodes_len = self.nodes.len();

                    // Update the split node
                    self.nodes[split_info.index].variance = split_info.new_variance;
                    self.nodes[split_info.index].info = split_info.new_stem;

                    debug!(
                        "Left: {}, right: {}, depth: {}",
                        split_info.left_indexes.len(),
                        split_info.right_indexes.len(),
                        split_info.left_node.depth
                    );
                    debug!(
                        "variance: {} {}\n value: {} {}",
                        split_info.left_node.variance,
                        split_info.right_node.variance,
                        split_info.left_node.value,
                        split_info.right_node.value
                    );
                    // add Left child
                    split_info.left_node.index = curr_nodes_len;
                    self.nodes.push(split_info.left_node);
                    current_nodes.push(curr_nodes_len);
                    node_sample_index.insert(curr_nodes_len, split_info.left_indexes);
                    debug!("Added {} {}", curr_nodes_len, curr_nodes_len + 1);
                    // add Right Child
                    split_info.right_node.index = curr_nodes_len + 1;
                    self.nodes.push(split_info.right_node);
                    current_nodes.push(curr_nodes_len + 1);
                    node_sample_index.insert(curr_nodes_len + 1, split_info.right_indexes);

                    //set left and right for parent node
                    if let NodeInfo::Stem {
                        ref mut feature,
                        ref mut param,
                        ref mut left,
                        ref mut right,
                    } = self.nodes[split_info.index].info
                    {
                        debug!("split on {}: {}", feature, param);
                        *left = curr_nodes_len;
                        *right = curr_nodes_len + 1;
                    } else {
                        panic!("expected stem as parent node, found leaf")
                    }
                }
            }
        }
    }
}

impl Learner for DecisionTree {
    fn fit(&mut self, x: &DataFrame, y: &DataFrame) {
        let features_order = get_df_sorted_perm(&x);

        self.build_model(features_order, x, y);
    }

    fn predict(&self, df: &DataFrame) -> DataFrame {
        if self.nodes.is_empty() {
            panic!("Model is not trained!");
        }
        let pred: Vec<V> = (0..df.rows())
            .into_par_iter()
            .map(|row_index| {
                // start from root
                let mut current_node = 0 as usize;
                let mut value = self.nodes[current_node].value;
                loop {
                    match &self.nodes[current_node].info {
                        Stem {
                            ref feature,
                            ref param,
                            ref left,
                            ref right,
                        } => {
                            let row_val = df[[row_index, *feature]];
                            if row_val <= *param {
                                current_node = *left;
                            } else {
                                current_node = *right;
                            }
                            value = self.nodes[current_node].value;
                        }
                        NodeInfo::Leaf => {
                            break;
                        }
                    }
                }
                value
            })
            .collect();

        DataFrame::from_shape_vec((1, df.rows()), pred).unwrap()
    }
}
