import json
import pandas as pd


# functions for validating schemas

def load_input_schema(fname='input_schema.json'):
    with open(fname, mode='r') as f:
        return json.load(f)


def validate_table(df, table_name, schema):
    assert isinstance(df, pd.DataFrame), "input must be DataFrame"

    if table_name not in schema.keys():
        raise ValueError("table_name '{}' not found in JSON schema".format(table_name))

    target_schema = schema[table_name]
    output_columns = []

    for column_def in target_schema:

        # check column exists in dataframe
        if column_def['name'] not in df.columns.values:
            raise ValueError("missing column '{}'".format(column_def["name"]))

        # check column does not contain nulls:
        if column_def['nullable'] == 'false' and not df[column_def['name']].notnull().all():
            raise ValueError("Non-nullable column '{}' contains null values".format(column_def['name']))

        # check casting
        try:
            output_columns.append(df[column_def['name']].astype(column_def['dtype']))
        except Exception as e:
            cast_error_msg = e.args[0]
            raise ValueError("Invalid cast for column '{}' to type '{}', message: \n '{}'".format(
                column_def['name'],
                column_def['dtype'],
                cast_error_msg
            ))

        # return converted data
        return pd.concat(output_columns, axis=1)




