"""
Postmodeling parameters

This script contain the parameters Class that will be used across all the
postmodeling functions within the ModelEvaluator and ModelGroupEvaluator
classes. This class will be initialized using the 'postmodeling_parameters.yaml'
file

"""

import yaml
import json

import verboselogs, logging
logger = verboselogs.VerboseLogger(__name__)

class PostmodelParameters:
    '''
    PostmodelParameters reads all parameters from a 'yaml' file and store them
    in this object to be used in other functions. Different metrics can be
    passed to this object, by default it will reads from a
    'postmodeling_parameters.yaml', but an Audition config file can be passed
    and will parse from it the needed parameters
    '''
    def __init__(self, path_params):

        with open(path_params) as f:
            params = yaml.full_load(f)

        # Assign dict elements to Parameters object and flatten
        # thresholds
        self.__dict__.update(params)
        self.figsize = tuple(self.figsize)

        try:
            if self.audition_output_path is not None:
                with open(self.audition_output_path) as f:
                    json_models = json.load(f)

                list_models = [model for model_list in json_models.values()
                               for model in model_list]
                self.model_group_id = list_models

        except AttributeError:
            logger.exception(
                f'''No audition output file was defined. I will use the models
                defined in the {path_params} configuration file.'''
            )
