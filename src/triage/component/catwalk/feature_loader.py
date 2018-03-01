import pandas as pd
import logging
import pdb
from .features import class_map
from .features import officers_collate

log = logging.getLogger(__name__)


class FeatureLoader():

    def __init__(self, features, 
                       schema_name, 
                       blocks, 
                       labels_config, 
                       labels, 
                       labels_table, 
                       prediction_window, 
                       officer_past_activity_window,
                       timegated_feature_lookback_duration,
                       db_engine):
        '''
        Args:
            feature_blocks (dict): dictionary of feature blocks and list of features to use for the matrix
            schema_name (str) : name of the schema in the db where the features blocks tables are
            blocks (list): name of the feature blocks to use
            labels_config (dict): config file of the conditions for each label
            labels (dict): labels dictionary to use from the config file
            prediction_window (str) : prediction window to use for the label generation
            officer_past_activity_window (str): window for conditioning which officers to use given an as_of_date
        '''

        self.features = features
        self.schema_name = schema_name
        self.blocks = blocks
        self.labels_config = labels_config
        self.labels = labels
        self.labels_table = labels_table
        self.prediction_window = prediction_window
        self.officer_past_activity_window = officer_past_activity_window
        self.timegated_feature_lookback_duration = timegated_feature_lookback_duration
        self.db_engine = db_engine

        self.flatten_label_keys = [item for sublist in self.labels for item in sublist]

    def _block_tables_name(self, block_name):
        block_class = class_map.lookup_block(block_name,
                                     module=officers_collate,
                                     lookback_durations=self.timegated_feature_lookback_duration,
                                     n_cpus=1)

        list_prefix = [block_class.prefix_space_time_lookback,
                       block_class.prefix_sub,
                       block_class.prefix_agg,
                       block_class.prefix_space_time, 
                       block_class.prefix_post]
        
        return ['{prefix}_aggregation'.format(prefix=prefix) for prefix in list_prefix if prefix]       


    def features_list(self):
        return [feature for list_features in self.features_in_blocks().values() for feature in list_features]

    def features_in_blocks(self):
        
        features_in_blocks = {}
        features_missing = [] 
        for block in self.blocks:
            active_features = [key for key in self.features[block] if self.features[block][key] == True]
            block_tables = self._block_tables_name(block)
            for block_table in block_tables:
                if active_features:
                    query = (""" select * FROM public.get_active_block_features('{schema_name}',
                                                                                '{block_table}',
                                                                                 ARRAY{active_features},
                                                                                 ARRAY{timegated_feature_lookback_duration});"""
                             .format(schema_name=self.schema_name,
                                     block_table=block_table,
                                     active_features=active_features,
                                     timegated_feature_lookback_duration=self.timegated_feature_lookback_duration))
                    
                    result = self.db_engine.connect().execute(query)
                    result_dict = [dict(row) for row in result][0]
                    features_in_blocks[block_table] = result_dict['col_avaliable']
                    # keep going through the rest of features
                    active_features = result_dict['col_missing']

            if result_dict['col_missing']:
                features_missing += result_dict['col_missing']
        if not features_missing:
            log.debug('No features are missing')
        else:
            log.debug('These features are missing: {}'.format(features_missing))

        return features_in_blocks
         
    def _tree_conditions(self, nested_dict, parent=[], conditions=[]):
        '''
        Function that returns a list of conditions from the labels config file
        looping recursively through each tree
        Args:
            nested_dict (dict): dictionary for each of the keys in the labels_config file
            parent (list): use in the recursive function to append the parent to each tree
            conditions (list): use in the recursive mode to append all the conditions
        '''
        if isinstance(nested_dict, dict):
            column = nested_dict['COLUMN']
            for value in nested_dict['VALUES']:
                parent_temp = parent.copy()
                if isinstance(value, dict):
                    for key in value.keys():
                        parent_temp.append('{col}:{val}'.format(col=column, val=key))
                        self._tree_conditions(value[key], parent_temp, conditions)
                else:
                    parent_temp.append('{col}:{val}'.format(col=column, val=value))
                    conditions.append('{{{parent_temp}}}'.format(parent_temp=",".join(parent_temp)))
        return conditions

    def _get_event_type_columns(self, nested_dict, list_events=[]):
        if isinstance(nested_dict, dict):
            list_events.append(nested_dict['COLUMN'])
            for val in nested_dict['VALUES']:
                if isinstance(val, dict):
                    for key in val.keys():
                        self._get_event_type_columns(val[key], list_events)
        return list_events

    def get_query_labels(self, as_of_dates_to_use):
        '''
        '''

        # SUBQUERIES of arrays of conditions
        sub_query = []
        event_type_columns = set()
        for key in self.flatten_label_keys:
            condition = key.lower()
            list_conditions = self._tree_conditions(self.labels_config[key], parent=[], conditions=[])
            sub_query.append(" {condition}_table as "
                            "    ( SELECT  "
                            "          unnest(ARRAY{list_conditions}) as {condition}_condition )"
                            .format(condition=condition,
                                    list_conditions=list_conditions))
            # event type
            event_type_columns.update(self._get_event_type_columns(self.labels_config[key], []))

        # JOIN subqueries
        sub_queries = ", ".join(sub_query)
        sub_queries = ("WITH {sub_queries}, "
                       " all_conditions as "
                       "    (SELECT * "
                       "     FROM {cross_joins})"
                       .format(sub_queries=sub_queries,
                               cross_joins=" CROSS JOIN ".join([key.lower() + '_table' for key in self.flatten_label_keys])))

        # CREATE AND AND OR CONDITIONS
        and_conditions = []
        for and_labels in self.labels:
            or_conditions = []
            for label in and_labels:
                or_conditions.append("event_type_array::text[] @> {key}_condition::text[]".format(key=label.lower()))
            and_conditions.append(" OR ".join(or_conditions))
        conditions = " AND ".join('({and_condition})'.format(and_condition=and_condition) for and_condition in and_conditions)

        # QUERY OF AS OF DATES
        query_as_of_dates = (" as_of_dates as ( "
                            "select unnest(ARRAY{as_of_dates}::timestamp[]) as as_of_date) "
                            .format(as_of_dates=as_of_dates_to_use))

        # DATE FILTER
        query_filter = ("group_events as ( "
                        "SELECT officer_id,  "
                        "       event_id, "
                        "       array_agg(event_type::text ||':'|| value::text ORDER BY 1) as event_type_array, "
                        "       min(event_datetime) as min_date, "
                        "       max(event_datetime) filter (where event_type in "
                        "                          (SELECT unnest(ARRAY{event_types}))) as max_date "
                        "FROM features.{labels_table}  "
                        "GROUP BY officer_id, event_id  "
                        "), date_filter as ( "
                        " SELECT  officer_id, "
                        "        as_of_date, "
                        "        event_type_array "
                        " FROM group_events "
                        " JOIN  as_of_dates ON "
                        " min_date > as_of_date and max_date < as_of_date + INTERVAL '{prediction_window}') "
                        .format(event_types=list(event_type_columns),
                                labels_table=self.labels_table,
                                prediction_window=self.prediction_window))

        query_select_labels = (" labels as ( "
                               "  SELECT officer_id, "
                               "        as_of_date, "
                               "        1 as outcome "
                               " FROM date_filter "
                               " JOIN all_conditions ON "
                               "   {conditions} "
                               " GROUP by as_of_date, officer_id)"
                               .format(conditions=conditions))

        # CONCAT all parts of query
        query_labels = ("{sub_queries}, "
                        "{as_of_dates}, "
                        "{query_filter}, "
                        "{query_select}".format(sub_queries=sub_queries,
                                                as_of_dates=query_as_of_dates,
                                                query_filter=query_filter,
                                                query_select=query_select_labels))
        return query_labels

    def get_dataset(self, as_of_dates_to_use):
        features_in_blocks = self.features_in_blocks()
        # Read labels master 
        complete_df = self.get_master_labels(as_of_dates_to_use)

        #loop through every table in blocks
        for table_name, features in features_in_blocks.items():
            log.info('Joining table {}!'.format(table_name))
            features_coalesce = ", ".join(['coalesce("{0}",0) as {0}'.format(feature) for feature in features])
             
            # table with no date
            if 'ND' in table_name:
                query = ("""SELECT officer_id,
                                   {features_coalesce}
                            FROM {schema}."{table_name}"
                             WHERE officer_id is not null; """
                                            .format(features_coalesce=features_coalesce,
                                                                schema=self.schema_name,
                                                                table_name=table_name))                                    
            else:
                query = ("""SELECT officer_id,
                                   as_of_date::timestamp,
                                  {features_coalesce}
                            FROM {schema}."{table_name}"
                            WHERE as_of_date in (
                                SELECT unnest(ARRAY{as_of_dates}::DATE[]))
                             AND officer_id is not null
                                ;""".format(features_coalesce=features_coalesce,
                                                 schema=self.schema_name,
                                                 table_name=table_name,
                                                 as_of_dates=as_of_dates_to_use))
            # Get the data
            db_conn = self.db_engine.raw_connection()
            cur = db_conn.cursor(name='cursor_for_loading_matrix')
            cur.execute(query)
            table = cur.fetchall()

            # Get column names
            col_names = []
            for desc in cur.description:
                col_names.append(desc[0])  

            # To pandas df
            table = pd.DataFrame(table)
            table.columns = col_names
            db_conn.close()
            
            if 'ND' in table_name:
                 complete_df = complete_df.merge(table, on='officer_id', how='left')
            else:
                 complete_df = complete_df.merge(table, on=['officer_id','as_of_date'], how='left')

        #Set index
        complete_df = complete_df.set_index('officer_id')

        # Zero imputation
        complete_df.fillna(0, inplace=True)

        # labels at last
        cols = complete_df.columns.tolist()
        cols.insert(len(cols), cols.pop(cols.index('outcome')))
        complete_df = complete_df.reindex(columns=cols)  

        log.info('length of data_set: {}'.format(len(complete_df)))
        log.info('as of dates used: {}'.format(complete_df['as_of_date'].unique()))
        log.info('number of officers with adverse incident: {}'.format(complete_df['outcome'].sum() ))
        return complete_df

    def get_query_features(self):
        table_names = [x for block in self.blocks for x in  self._block_tables_name(block)]  

        #[item for sublist in self.labels for item in sublist]
        # seperate the tables by block that have a date column or not
        table_names_no_date = [x for x in table_names if 'ND' in x]
        table_names_with_date = [x for x in table_names if x not in set(table_names_no_date)]

        # convert features to string for querying while replacing NULL values with ceros in sql
        features_coalesce = ", ".join(['coalesce("{0}",0) as {0}'.format(feature) for feature in self.features_list()])

        query = ""
        if len(table_names) > 0:
            if table_names_with_date:
                query = (""" SELECT officer_id,
                                    as_of_date,
                                    {features_coalesce}
                              FROM {schema}."{block_table}" """.format(features_coalesce=features_coalesce,
                                                                       schema=self.schema_name,
                                                                       block_table= table_names_with_date[0]))
                if len(table_names_with_date) > 1:
                    table_names_with_date = table_names_with_date[1:]
                    for table_name in table_names_with_date:
                        query += (""" FULL OUTER JOIN {schema}."{block_table}" 
                                          USING (officer_id, as_of_date)""".format(schema=self.schema_name,
                                                                                    block_table= table_name))
   
            # check if in the first loop above a table was added
            if len(query) == 0:
                query = (""" SELECT officer_id,
                                    {features_coalesce}
                            FROM {schema}."{block_table}" """.format(features_coalesce=features_coalesce,
                                                                     schema=self.schema_name, 
                                                                     block_table=table_names_no_date[0]))
                if len(table_names_no_date) > 1:
                    table_names_no_date = table_names_no_date[1:]
                    for table_name in table_names_no_date:
                        query += (""" FULL OUTER JOIN {schema}."{block_table}" 
                                           USING (officer_id)""".format(schema=self.schema_name,
                                                                        block_table= table_name)) 
            # Filter by date
            query += """ WHERE as_of_date in ( SELECT as_of_date from as_of_dates) """
            subquery = """ features as ({query})""".format(query=query)

            return subquery
        
    def get_master_labels(self, as_of_dates_to_use):
        '''
        Returns master list of labels for specific as of dates
        '''

        # We only want to train and test on officers that have been active (any logged activity in events_hub)
        # NOTE: it uses the feature_labels created in query_labels         
        active_subquery = ( " officers AS (  "
                         "       SELECT officer_id "
                         "       FROM staging.officers_hub "
                         " ), active AS ( "
                         "       SELECT officer_id, as_of_date "
                         "       FROM as_of_dates as d "
                         "       CROSS JOIN officers as off, "
                         "           LATERAL "
                         "                (SELECT 1 "
                         "                 FROM staging.events_hub e "
                         "                 WHERE off.officer_id = e.officer_id "
                         "                 AND e.event_datetime + INTERVAL '{window}' > d.as_of_date "
                         "                 AND e.event_datetime <= d.as_of_date "
                         "                    LIMIT 1 ) sub_activity, "
                         "            LATERAL "
                         "                (SELECT 1 "
                         "                 FROM staging.officer_roles r "
                         "                 WHERE off.officer_id = r.officer_id "
                         "                 AND r.job_start_date <= d.as_of_date "
                         "                 AND sworn_flag = 1 "
                         "                 LIMIT 1) sub_sworn )"
                         .format(window=self.officer_past_activity_window))

        query_master_labels = (" {labels_subquery}, "
                               " {active_subquery} "
                               " SELECT officer_id, "
                               "        as_of_date, "
                               "        coalesce(outcome,0) as outcome "
                               " FROM active "
                               " LEFT JOIN labels "
                               " USING (as_of_date, officer_id) "
                               .format(labels_subquery=self.get_query_labels(as_of_dates_to_use),
                                       active_subquery=active_subquery))
        
        db_conn = self.db_engine.raw_connection()
        cur = db_conn.cursor(name='cursor_for_loading_matrix')
        cur.execute(query_master_labels)
        labels = cur.fetchall()

        # Get column names
        col_names = []
        for desc in cur.description:
            col_names.append(desc[0])

        # To pandas df
        labels = pd.DataFrame(labels)
        labels.columns = col_names
        db_conn.close()
        return labels

    def get_dataset_old(self, as_of_dates_to_use):
        '''
        This function returns dataset and labels to use for training / testing
        It is splitted in two queries:
            - features_subquery: which joins the features table with labels table
            - query_active: using the first table created in query_labels, and returns it only
                            for officers that are have any activity given the officer_past_activity_window

        '''
        labels = self.get_master_labels(as_of_dates_to_use)
        pdb.set_trace()
        features_list_string = ", ".join(['{}'.format(feature) for feature in self.features_list()])

        # JOIN FEATURES AND LABELS
        query_features_labels = (" {labels_subquery}, "
                                 " {features_subquery}, " 
                                 " features_labels AS ( "
                                 "    SELECT officer_id, "
                                 "           as_of_date, "
                                 "           {features_list}, "
                                 "           coalesce(outcome,0) as outcome "
                                 "    FROM features "
                                 "    LEFT JOIN labels "
                                 "    USING (as_of_date, officer_id)) "
                                 .format(labels_subquery=self.get_query_labels(as_of_dates_to_use),
                                         features_subquery=self.get_query_features(),
                                         features_list=features_list_string))

        # We only want to train and test on officers that have been active (any logged activity in events_hub)
        # NOTE: it uses the feature_labels created in query_labels
        query_active =  (""" SELECT officer_id, as_of_date, {features}, outcome """
                        """ FROM features_labels as f, """
                        """        LATERAL """
                        """          (SELECT 1 """
                        """           FROM staging.events_hub e """
                        """           WHERE f.officer_id = e.officer_id """
                        """           AND e.event_datetime + INTERVAL '{window}' > f.as_of_date """
                        """           AND e.event_datetime <= f.as_of_date """
                        """            LIMIT 1 ) sub; """
                        .format(features=features_list_string,
                                window=self.officer_past_activity_window))

        # join both queries together and load data
        query = (query_features_labels + query_active)

        # Get the data
        db_conn = self.db_engine.raw_connection()
        cur = db_conn.cursor(name='cursor_for_loading_matrix')
        cur.execute(query)
        matrix = cur.fetchall()

        # Get column names
        col_names = []
        for desc in cur.description:
            col_names.append(desc[0])  

        # To pandas df
        matrix_df = pd.DataFrame(matrix)
        matrix_df.columns = col_names
        db_conn.close()

        #all_data = pd.read_sql(query, con=db_conn)

        ## TODO: remove all zero value columns
        #all_data = all_data.loc[~(all_data[features_list]==0).all(axis=1)]

        log.info('length of data_set: {}'.format(len(matrix_df)))
        log.info('as of dates used: {}'.format(matrix_df['as_of_date'].unique()))
        log.info('number of officers with adverse incident: {}'.format(matrix_df['outcome'].sum() ))
        return matrix_df

