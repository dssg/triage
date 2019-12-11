create extension postgis;

create extension postgis_raster;
create extension postgis_topology;
create extension postgis_sfcgal;



create extension if not exists fuzzystrmatch;
create extension if not exists unaccent;
create extension if not exists pg_trgm;
create extension if not exists bloom;

create extension if not exists citext;

create extension if not exists cube;

create extension if not exists file_fdw;
create extension if not exists postgres_fdw;

create extension if not exists earthdistance;
