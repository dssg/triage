CREATE SCHEMA postgis;

ALTER DATABASE food SET search_path=public, postgis, contrib;

CREATE EXTENSION postgis SCHEMA postgis;

-- Enable topology
CREATE EXTENSION postgis_topology;

-- Enable PostGIS Advanced 3D and other geoprocessing algorithms
CREATE EXTENSION postgis_sfcgal;

CREATE EXTENSION pgrouting;
