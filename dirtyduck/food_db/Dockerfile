# DirtyDuck Food Inspections Database
# Modernized: PostgreSQL 16 + PostGIS 3

FROM postgres:16

# Install PostGIS and dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        postgresql-16-postgis-3 \
        postgresql-16-postgis-3-scripts \
        postgis \
        bzip2 && \
    rm -rf /var/lib/apt/lists/*

# Chicago Food Inspections dataset
# From https://data.cityofchicago.org/api/views/4ijn-s7e5/rows.csv?accessType=DOWNLOAD
# Only rows between 2014-2017 (inclusive) are included
COPY inspections_2014_2017.csv.bz2 /tmp/

# Database initialization scripts (run in order)
COPY 00_create_extensions.sql /docker-entrypoint-initdb.d/
COPY 01_create_inspections_table.sql /docker-entrypoint-initdb.d/
COPY 02_create_cleaned_inspections_table.sql /docker-entrypoint-initdb.d/
COPY 03_create_violations_table.sql /docker-entrypoint-initdb.d/
COPY 04_create_semantic_tables.sql /docker-entrypoint-initdb.d/

# Set proper permissions
RUN chown postgres:postgres /docker-entrypoint-initdb.d/*.sql
