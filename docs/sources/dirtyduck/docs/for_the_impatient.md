# Appendix: For the impatient

If you want to skip all the cleansing and transformation and deep directly into `triage` you can execute the following *inside bastion*:

```sh
     curl "https://data.cityofchicago.org/api/views/4ijn-s7e5/rows.csv?accessType=DOWNLOAD" > data/inspections.csv

     psql ${DATABASE_URL} -c "\copy raw.inspections FROM '/data/inspections.csv' WITH HEADER CSV"

     psql ${DATABASE_URL} < /sql/create_cleaned_inspections_table.sql

     psql ${DATABASE_URL} < /sql/create_violations_table.sql

     psql ${DATABASE_URL} < /sql/create_semantic_tables.sql
```

If everything works, you should end with two new schemas: `cleaned` and `semantic`.

You could check that (from `psql`) With

```sql
\dn
```

| List of schemas |                     |
|--------------- |------------------- |
| Name            | Owner               |
| cleaned         | food<sub>user</sub> |
| postgis         | food<sub>user</sub> |
| public          | postgres            |
| raw             | food<sub>user</sub> |
| semantic        | food<sub>user</sub> |

Now you can continue to the introduction to triage section.
