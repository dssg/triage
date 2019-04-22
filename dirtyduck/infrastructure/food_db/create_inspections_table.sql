create schema if not exists raw;

create table raw.inspections (
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
