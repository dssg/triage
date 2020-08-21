/*
  Uses hash partitioning on the {train_results, test_results}.predictions  tables
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
)   partition by hash(model_id);

create table test_results.predictions_1 partition of test_results.predictions for values with (modulus 30, remainder 0);
create table test_results.predictions_2 partition of test_results.predictions for values with (modulus 30, remainder 1);
create table test_results.predictions_3 partition of test_results.predictions for values with (modulus 30, remainder 2);
create table test_results.predictions_4 partition of test_results.predictions for values with (modulus 30, remainder 3);
create table test_results.predictions_5 partition of test_results.predictions for values with (modulus 30, remainder 4);
create table test_results.predictions_6 partition of test_results.predictions for values with (modulus 30, remainder 5);
create table test_results.predictions_7 partition of test_results.predictions for values with (modulus 30, remainder 6);
create table test_results.predictions_8 partition of test_results.predictions for values with (modulus 30, remainder 7);
create table test_results.predictions_9 partition of test_results.predictions for values with (modulus 30, remainder 8);
create table test_results.predictions_10 partition of test_results.predictions for values with (modulus 30, remainder 9);

create table test_results.predictions_11 partition of test_results.predictions for values with (modulus 30, remainder 10);
create table test_results.predictions_12 partition of test_results.predictions for values with (modulus 30, remainder 11);
create table test_results.predictions_13 partition of test_results.predictions for values with (modulus 30, remainder 12);
create table test_results.predictions_14 partition of test_results.predictions for values with (modulus 30, remainder 13);
create table test_results.predictions_15 partition of test_results.predictions for values with (modulus 30, remainder 14);
create table test_results.predictions_16 partition of test_results.predictions for values with (modulus 30, remainder 15);
create table test_results.predictions_17 partition of test_results.predictions for values with (modulus 30, remainder 16);
create table test_results.predictions_18 partition of test_results.predictions for values with (modulus 30, remainder 17);
create table test_results.predictions_19 partition of test_results.predictions for values with (modulus 30, remainder 18);
create table test_results.predictions_20 partition of test_results.predictions for values with (modulus 30, remainder 19);

create table test_results.predictions_21 partition of test_results.predictions for values with (modulus 30, remainder 20);
create table test_results.predictions_22 partition of test_results.predictions for values with (modulus 30, remainder 21);
create table test_results.predictions_23 partition of test_results.predictions for values with (modulus 30, remainder 22);
create table test_results.predictions_24 partition of test_results.predictions for values with (modulus 30, remainder 23);
create table test_results.predictions_25 partition of test_results.predictions for values with (modulus 30, remainder 24);
create table test_results.predictions_26 partition of test_results.predictions for values with (modulus 30, remainder 25);
create table test_results.predictions_27 partition of test_results.predictions for values with (modulus 30, remainder 26);
create table test_results.predictions_28 partition of test_results.predictions for values with (modulus 30, remainder 27);
create table test_results.predictions_29 partition of test_results.predictions for values with (modulus 30, remainder 28);
create table test_results.predictions_30 partition of test_results.predictions for values with (modulus 30, remainder 29);


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
)   partition by hash(model_id);


create table train_results.predictions_1 partition of train_results.predictions for values with (modulus 30, remainder 0);
create table train_results.predictions_2 partition of train_results.predictions for values with (modulus 30, remainder 1);
create table train_results.predictions_3 partition of train_results.predictions for values with (modulus 30, remainder 2);
create table train_results.predictions_4 partition of train_results.predictions for values with (modulus 30, remainder 3);
create table train_results.predictions_5 partition of train_results.predictions for values with (modulus 30, remainder 4);
create table train_results.predictions_6 partition of train_results.predictions for values with (modulus 30, remainder 5);
create table train_results.predictions_7 partition of train_results.predictions for values with (modulus 30, remainder 6);
create table train_results.predictions_8 partition of train_results.predictions for values with (modulus 30, remainder 7);
create table train_results.predictions_9 partition of train_results.predictions for values with (modulus 30, remainder 8);
create table train_results.predictions_10 partition of train_results.predictions for values with (modulus 30, remainder 9);

create table train_results.predictions_11 partition of train_results.predictions for values with (modulus 30, remainder 10);
create table train_results.predictions_12 partition of train_results.predictions for values with (modulus 30, remainder 11);
create table train_results.predictions_13 partition of train_results.predictions for values with (modulus 30, remainder 12);
create table train_results.predictions_14 partition of train_results.predictions for values with (modulus 30, remainder 13);
create table train_results.predictions_15 partition of train_results.predictions for values with (modulus 30, remainder 14);
create table train_results.predictions_16 partition of train_results.predictions for values with (modulus 30, remainder 15);
create table train_results.predictions_17 partition of train_results.predictions for values with (modulus 30, remainder 16);
create table train_results.predictions_18 partition of train_results.predictions for values with (modulus 30, remainder 17);
create table train_results.predictions_19 partition of train_results.predictions for values with (modulus 30, remainder 18);
create table train_results.predictions_20 partition of train_results.predictions for values with (modulus 30, remainder 19);

create table train_results.predictions_21 partition of train_results.predictions for values with (modulus 30, remainder 20);
create table train_results.predictions_22 partition of train_results.predictions for values with (modulus 30, remainder 21);
create table train_results.predictions_23 partition of train_results.predictions for values with (modulus 30, remainder 22);
create table train_results.predictions_24 partition of train_results.predictions for values with (modulus 30, remainder 23);
create table train_results.predictions_25 partition of train_results.predictions for values with (modulus 30, remainder 24);
create table train_results.predictions_26 partition of train_results.predictions for values with (modulus 30, remainder 25);
create table train_results.predictions_27 partition of train_results.predictions for values with (modulus 30, remainder 26);
create table train_results.predictions_28 partition of train_results.predictions for values with (modulus 30, remainder 27);
create table train_results.predictions_29 partition of train_results.predictions for values with (modulus 30, remainder 28);
create table train_results.predictions_30 partition of train_results.predictions for values with (modulus 30, remainder 29);


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
