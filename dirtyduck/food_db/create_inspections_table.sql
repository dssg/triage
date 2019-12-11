create schema if not exists raw;

drop table if exists raw.inspections;
create table if not exists raw.inspections (
       inspection varchar not null,
       DBA_Name varchar,
       AKA_Name varchar,
       license_Num decimal,
       facility_type varchar,
       risk varchar,
       address varchar,
       city varchar,
       state varchar,
       zip varchar,
       date date,
       type varchar,
       results varchar,
       violations varchar,
       latitude decimal,
       longitude decimal,
       location varchar
);

copy raw.inspections from program 'bzcat /tmp/inspections_2014_2017.csv.bz2' HEADER CSV QUOTE '"';
