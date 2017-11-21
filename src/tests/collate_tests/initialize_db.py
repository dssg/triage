import yaml
import os
from os import path, system

# this is hacky, but it's easiest to simply change directories to ensure relative paths work:
os.chdir(path.dirname(path.abspath(__file__)))
# We'll shell out to `psql`, so set the environment variables for it:
with open("config/database.yml") as f:
    for k,v in yaml.load(f).items():
        os.environ['PG' + k.upper()] = v if v else ""
# And create the table from the csv file with psql
system("""psql -c "DROP TABLE IF EXISTS food_inspections;" """)
system("""csvsql --no-constraints --table food_inspections < food_inspections_subset.csv | psql """)
system("""psql -c "\copy food_inspections FROM 'food_inspections_subset.csv' WITH CSV HEADER;" """)
system("""psql -c "CREATE INDEX ON food_inspections(license_no, inspection_date)" """)

# create a state table for license/date
system("""psql -c "DROP TABLE IF EXISTS inspection_states;" """)
sql = """CREATE TABLE inspection_states AS (
SELECT license_no, date
FROM (SELECT DISTINCT license_no FROM food_inspections) a
CROSS JOIN (SELECT DISTINCT inspection_date as date FROM food_inspections) b
)""".replace('\n', ' ')
system("""psql -c "%s" """ % sql)
system("""psql -c "CREATE INDEX ON inspection_states(license_no, date)" """)

# create an alternate state table with a different date column
system("""psql -c "DROP TABLE IF EXISTS inspection_states_diff_colname;" """)
system("""psql -c "CREATE TABLE inspection_states_diff_colname AS select license_no, date as aggregation_date from inspection_states" """)
system("""psql -c "CREATE INDEX ON inspection_states_diff_colname(license_no, aggregation_date)" """)

# create a state table for licenseo only
system("""psql -c "DROP TABLE IF EXISTS all_licenses;" """)
sql = """CREATE TABLE all_licenses AS (
SELECT DISTINCT license_no FROM food_inspections
)""".replace('\n', ' ')
system("""psql -c "%s" """ % sql)
system("""psql -c "CREATE INDEX ON all_licenses(license_no)" """)
