create schema if not exists raw;

drop table if exists raw.inspections;
create table if not exists raw.inspections (
       inspection text not null,
       DBA_Name text,
       AKA_Name text,
       license_Num decimal,
       facility_type text,
       risk text,
       address text,
       city text,
       state text,
       zip text,
       date date,
       type text,
       results text,
       violations text,
       latitude decimal,
       longitude decimal,
       location text
);

copy raw.inspections from program 'bzcat /tmp/inspections_2014_2017.csv.bz2' HEADER CSV QUOTE '"';
