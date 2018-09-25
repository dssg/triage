"""
Postmodeling parameters

This script contain the parameters Class that will be used across all the
postmodeling functions within the ModelEvaluator and ModelGroupEvaluator
classes. This class will be initialized using the 'postmodeling_parameters.yaml' 
file.abs

"""

import yaml


class PostmodelParameters(object):
    '''
    PostmodelParameters reads all parameters from a 'yaml' file and store them
    in this object to be used in other functions. Different metrics can be
    passed to this object, by default it will reads from a
    'postmodeling_parameters.yaml', but an Audition config file can be passed
    and will parse from it the needed parameters
    '''
    def __init__(self, path_params):

        with open(path_params) as f:
            params = yaml.load(f)
        self.__dict__.update(params)



