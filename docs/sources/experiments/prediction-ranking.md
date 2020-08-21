# Prediction Ranking

The predictions tables in the `train_results` and `test_results`
schemas contain several different flavors of rankings, covering
absolute vs percentile ranking and whether or not ties exist.

## Ranking columns

| Column name | Behavior |
| ----------- | ------- |
| rank_abs_with_ties | Absolute ranking, with ties. Ranks will skip after a set of ties, so if two entities are tied at rank 3, the next entity after them will have rank 5. |
| rank_pct_with_ties | Percentile ranking, with ties. Percentiles will skip after a set of ties, so if two entities out of ten are tied at 0.1 (tenth percentile), the next entity after them will have 0.3 (thirtieth percentile). At most five decimal places. |
| rank_abs_no_ties | Absolute ranking, with no ties. Ties are broken according to a configured choice: 'best', 'worst', or 'random', which is recorded in the `prediction_metadata` table |
| rank_pct_no_ties | Percentile ranking, with no ties. Ties are broken according to a configured choice: 'best', 'worst', or 'random', which is recorded in the `prediction_metadata` table. At most five decimal places. |


## Viewing prediction metadata

The `prediction_metadata` table contains information about how ties
were broken. There is one row per model/matrix combination. For each
model and matrix, it records:

- `tiebreaker_ordering` - The tiebreaker ordering rule (e.g. 'random',
  'best', 'worst') used for the corresponding predictions.
- `random_seed` - The random seed, if 'random' was the ordering
  used. Otherwise None
- `predictions_saved` - Whether or not predictions were saved. If it's
  false, you won't expect to find any predictions, but the row is
  inserted as a record that the prediction was performed.

There is one `prediction_metadata` table in each of the
`train_results`, `test_results` schemas (in other words, wherever
there is a companion `predictions` table).



## Subsequent runs

If you run Triage Experiments with `replace=False`, and you change
nothing except for the `rank_tiebreaker` in experiment config, ranking
will be redone and the row in `prediction_metadata` updated. You don't
have to run a full experiment if that's all you want to do; you could
follow the directions for backfilling ranks above, which will redo the
ranking for an individual model/matrix pair. However, changing the
`rank_tiebreaker` in experiment config and re-running the experiment
is a handy way of redoing all of them if that's what is useful.
