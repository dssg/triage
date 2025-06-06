import pandas as pd
import logging



class ModelGroupComparison:
    
    def __init__(self, model_group_ids, engine):
        """
        Initialize the ModelGroupComparison with model group IDs and a database engine.
        
        :param model_group_ids: List of model group IDs to compare.
        :param engine: Database engine for executing SQL queries.
        """
        
        self.model_group_ids = model_group_ids
        self.engine = engine
        