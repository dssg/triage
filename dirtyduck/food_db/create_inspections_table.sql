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
       location varchar,
       historical_wards varchar,
       zip_codes varchar,
       community_areas varchar,
       census_tracts varchar,
       wards varchar
);

comment on column raw.inspections.historical_wards is 'Historical wards 2003-2015';
