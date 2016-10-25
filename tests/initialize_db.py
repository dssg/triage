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
