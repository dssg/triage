select
    entity_id,
    bool_or(outcome::bool)::integer as outcome
from events
where outcome_date >= '{as_of_date}'
    and outcome_date < '{as_of_date}'::timestamp + interval '{label_timespan}'
group by entity_id
