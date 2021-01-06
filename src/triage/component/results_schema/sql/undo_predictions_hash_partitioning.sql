/*
  Reverts hash partitioning on the {train_results, test_results}.predictions  tablesa
 */



alter table test_results.predictions rename to predictions_old;


create table test_results.predictions (
  model_id integer not null,
  entity_id bigint not null,
  as_of_date timestamp not null,
  score numeric(6,5) ,
  label_value integer ,
  rank_abs_no_ties integer ,
  rank_abs_with_ties integer ,
  rank_pct_no_ties numeric(6,5) ,
  rank_pct_with_ties numeric(6,5) ,
  matrix_uuid text,
  test_label_timespan interval
);

insert into test_results.predictions
select model_id,
       entity_id,
       as_of_date,
       score,
       label_value,
       rank_abs_no_ties,
       rank_abs_with_ties,
       rank_pct_no_ties,
       rank_pct_with_ties,
       matrix_uuid,
       test_label_timespan
  from test_results.predictions_old;

drop table if exists test_results.predictions_old cascade;

alter table test_results.predictions
  add primary key (model_id, entity_id, as_of_date);

alter table test_results.predictions
  add constraint test_results_predictions_matrix_uuid_fkey
  foreign key (matrix_uuid)
  references triage_metadata.matrices(matrix_uuid);

alter table test_results.predictions
  add constraint test_results_predictions_model_id_fkey
  foreign key (model_id)
  references triage_metadata.models(model_id);

alter table train_results.predictions rename to predictions_old;

create table train_results.predictions (
  model_id integer not null,
  entity_id bigint not null,
  as_of_date timestamp not null,
  score numeric(6,5) ,
  label_value integer ,
  rank_abs_no_ties integer ,
  rank_abs_with_ties integer ,
  rank_pct_no_ties numeric(6,5) ,
  rank_pct_with_ties numeric(6,5) ,
  matrix_uuid text,
  test_label_timespan interval
);


insert into train_results.predictions
select model_id,
       entity_id,
       as_of_date,
       score,
       label_value,
       rank_abs_no_ties,
       rank_abs_with_ties,
       rank_pct_no_ties,
       rank_pct_with_ties,
       matrix_uuid,
       test_label_timespan
  from train_results.predictions_old;

drop table if exists train_results.predictions_old cascade;


alter table train_results.predictions
  add primary key (model_id, entity_id, as_of_date);

alter table train_results.predictions
  add constraint train_results_predictions_matrix_uuid_fkey
  foreign key (matrix_uuid)
  references triage_metadata.matrices(matrix_uuid);

alter table train_results.predictions
  add constraint train_results_predictions_model_id_fkey
  foreign key (model_id)
  references triage_metadata.models(model_id);
