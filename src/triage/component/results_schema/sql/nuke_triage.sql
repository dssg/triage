/*
  Function for wiping out all triage tables, schemas, functions and indexes.
  It also deletes results_schema_versions (from alembic)
  Useful  when ou try to start clean (again) and try to recover some
  previous errors
 */
create or replace function nuke_triage()
    returns text as $result$

    declare
    result text;
    query text;

    begin

    execute 'drop schema if exists triage_metadata cascade';
    raise notice 'triage_metadata deleted';
    execute 'drop schema if exists features cascade';
    raise notice 'features deleted';
    execute 'drop schema if exists train_results cascade';
    raise notice 'train_results deleted';
    execute 'drop schema if exists test_results cascade';
    raise notice 'test_results deleted';

    execute 'drop table if exists results_schema_versions';
    raise notice 'results_schema_versions deleted';

    execute 'drop function if exists get_model_group_id';
    raise notice 'get_model_group_id deleted';

    execute 'drop type if exists experimentrunstatus';
    raise notice 'experimentrunstatus type deleted';


      select into query
                  string_agg(
                    format('drop table %I cascade;', tablename), E'\n'
                  )
        from   pg_tables
       where  tablename ~ 'cohort_|labels_|ranks_';

      if query is not null then
        raise notice '%', query;
        execute query;
      else
        raise notice 'no  labels or states tables from triage found';
      end if;

      return 'triage was send to the oblivion. Long live to triage!';
    end;
$result$ language plpgsql;

comment on function nuke_triage () is 'Function for wiping out all triage tables, schemas, functions and indexes.
  It also deletes results_schema_versions (from alembic)
  Useful  when ou try to start clean (again) and try to recover some
  previous errors';
