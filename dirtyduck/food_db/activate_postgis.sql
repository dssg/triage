CREATE SCHEMA postgis;

ALTER DATABASE food SET search_path=public, postgis, contrib;

CREATE EXTENSION postgis;

-- Enable topology
CREATE EXTENSION postgis_topology;

-- Enable PostGIS Advanced 3D and other geoprocessing algorithms
CREATE EXTENSION postgis_sfcgal;

CREATE EXTENSION pgrouting;


create extension if not exists fuzzystrmatch;
create extension if not exists unaccent;
create extension if not exists pg_trgm;


create extension if not exists bloom;

create extension if not exists citext;

create extension if not exists cube;

create extension if not exists file_fdw;
create extension if not exists postgres_fdw;

create extension if not exists earthdistance;



