FROM postgres:12

## Installing PostGIS
RUN apt-get -y update \
    && apt-get -y  install wget \
    && wget --quiet -O - http://apt.postgresql.org/pub/repos/apt/ACCC4CF8.asc | apt-key add - \
    && apt-get -y update \
    && apt-get -y install postgresql-12-postgis-3 postgis postgresql-12-pgrouting bzip2

## Chigago Food Inspections
## From https://data.cityofchicago.org/api/views/4ijn-s7e5/rows.csv?accessType=DOWNLOAD
## Only the rows between the years 2014 and 2017 (inclusive) are included
COPY inspections_2014_2017.csv.bz2 /tmp

## DB setup
ADD 00_create_extensions.sql /docker-entrypoint-initdb.d/
ADD 01_create_inspections_table.sql /docker-entrypoint-initdb.d/
ADD 02_create_cleaned_inspections_table.sql /docker-entrypoint-initdb.d/
ADD 03_create_violations_table.sql /docker-entrypoint-initdb.d/
ADD 04_create_semantic_tables.sql /docker-entrypoint-initdb.d/

RUN chown postgres:postgres /docker-entrypoint-initdb.d/*.sql
