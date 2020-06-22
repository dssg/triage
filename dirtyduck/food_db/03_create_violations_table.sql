drop table if exists cleaned.violations cascade;

create table cleaned.violations as (
  select
    inspection::integer,
    license_num::integer,
    date::date,
    btrim(tuple[1]) as code,
    lower(btrim(tuple[2])) as description,
    lower(btrim(tuple[3])) as comment,
    (case
     when btrim(tuple[1]) = '' then NULL
     when btrim(tuple[1])::int between 1 and 14 then 'critical' -- From the documentation
     when btrim(tuple[1])::int between 15 and 29  then 'serious'
     else 'minor'
     end
    ) as severity from
                      (
                        select
                          inspection,
                          license_num,
                          date,
                          regexp_split_to_array(   -- Create an array we will split the code, description, comment
                                                regexp_split_to_table( -- Create a row per each comment we split by |
                                                                      coalesce(            -- If there isn't a violation add '- Comments:'
                                                                               regexp_replace(violations, '[\n\r]+', '', 'g' )  -- Remove line breaks
                                                                               , '- Comments:')
                                                                               , '\|')  -- Split the violations
                                                                               , '(?<=\d+)\.\s*|\s*-\s*Comments:')  -- Split each violation in three
                                                                                                                    -- , '\.\s*|\s*-\s*Comments:')  -- Split each violation in three (Use this if your postgresql is kind off old
                            as tuple
                          from raw.inspections
                         where results in ('Fail', 'Pass', 'Pass w/ Conditions') and license_num is not null
                      ) as t
);
