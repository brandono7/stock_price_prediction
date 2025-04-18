# XGBoost


## File Directory

```sh
stock_price_prediction
│
└───xgboost
    │   xgboost_rolling_window.ipynb
    │   xgboost.md
    │   xgboost_base.ipynb
    │   xgboost_indiv.ipynb
    │   xgboost_preprocessing.ipynb
    │
    └───datasets
            train.csv
            val.csv
            test.csv
```
## Optimisation
Our tuning goal is to find the sweet spot in the bias–variance trade‑off: trees must be deep enough to capture real patterns in lagged price features (Open, High, Low, Close, Volume) without overfitting random noise. 

We therefore focus on capacity‑controlling hyperparameters:
- `max_depth` and `min_child_weight` to limit tree size and leaf support
- `gamma` to require meaningful gain before splitting
- `subsample`, `colsample_bytree` (both row and feature subsampling) to decorrelate trees. 

We also include regularization terms (`reg_alpha`, `reg_lambda`) to dampen overly large leaf weights, and a `learning_rate` to scale each tree’s contribution, paired with `n_estimators` to set the maximum number of boosting rounds.

Because exhaustively searching a high‑dimensional space is prohibitively slow, we first employ RandomizedSearchCV over wide, practitioner‑informed ranges:
- tree depths 3–9
- subsampling from 0.6 to 1.0, 
- regularization from 0 to 10

By randomly sampling 30 combinations and evaluating each with three cross‑validation—all on the GPU—we quickly identify promising regions of the hyperparameter space with minimal compute.

Having narrowed our search to a “neighborhood” around the best random results, we then run a GridSearchCV over a fine grid (±1 in depth, ±0.1 in subsampling, etc.). This second pass hones in on the precise settings that maximize validation accuracy, again leveraging GPU acceleration to keep each fold fast.

Finally, even the best n_estimators remains unknown until we see diminishing returns in model fit. We therefore perform a final training run with a generous upper bound on trees (e.g. 10 000) and attach an early‑stopping callback monitoring validation log‑loss: if no improvement occurs over 30 consecutive rounds, boosting halts. This dynamic pruning both prevents overfitting and ensures we use just enough trees to capture genuine signal.


## TODO:
- Add more features
- Rolling windows/Sliding window 
    - (1 day, 3 days 5 days, 1 week, 2 weeks, 1 month, 3 month, 6 months, 1 year)
- Indiviual Stocks
- Misc Optimization