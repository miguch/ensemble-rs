# ensembles-rs

A Rust crate implementing regression decision tree, Gradient Boosting and Random Forest ensemble.

## Bins

**cross validation**:

```
cargo run --bin cv
```

**predict test set**:

```
cargo run --bin predict
```

**test parallel performance**:

```shell
for ((i=1; i<=12 ; i++)) cargo run --release --bin parallel_performance $i 
```



## References

[Regression Tree](http://www.stat.cmu.edu/~cshalizi/350-2006/lecture-10.pdf)

[Efficient Determination of Dynamic Split Points in a Decision Tree](https://www.microsoft.com/en-us/research/publication/efficient-determination-dynamic-split-points-decision-tree/?from=http%3A%2F%2Fresearch.microsoft.com%2Fen-us%2Fum%2Fpeople%2Fdmax%2Fpublications%2Fsplits.pdf)

[Parallel Gradient Boosting Decision Trees](https://zhanpengfang.github.io/418home.html)

[LightGBM](https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree.pdf)

