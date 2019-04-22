CREATE SCHEMA postgis;

ALTER DATABASE food SET search_path=public, postgis, contrib;

CREATE EXTENSION postgis SCHEMA postgis;
CREATE EXTENSION pgrouting;
