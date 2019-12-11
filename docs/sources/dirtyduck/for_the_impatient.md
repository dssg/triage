# For the impatient

If you want to skip all the cleansing and transformation and deep directly into `triage` you can execute the following *inside bastion*:

```sh
     psql ${DATABASE_URL} -c "\copy raw.inspections from program 'curl "https://data.cityofchicago.org/api/views/4ijn-s7e5/rows.csv?accessType=DOWNLOAD"' HEADER CSV"

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
| cleaned         | food\_user |
| postgis         | food\_user |
| public          | postgres            |
| raw             | food\_user |
| semantic        | food\_user |
