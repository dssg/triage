"""
Postmodeling parameters

This script contain the parameters Class that will be used across all the
postmodeling functions within the ModelEvaluator and ModelGroupEvaluator
classes. This class will be initialized using the 'postmodeling_parameters.yaml' 
file

"""

import yaml
import json
import warnings

class  ThresholdIterator(dict):
    '''
    Take a dictionary of parameters and create an interator class that store
    the metric, and the parameters, for instance:
        {'rank_abs': [10, 20, 50], 'rank_pct': [10, 20, 50]}
        and yield a iterator that contains the key dict as identifier, and the
        values as iterators. 
    '''
    def __init__(self, dict):
        super().__init__(dict)

        pairs_thresholds = [(iter(v), k) for (k, v) in dict.items()]
        self.elements = lst = [] 
        for param, metric in pairs_thresholds:
            for threshold in param:
                lst.append((metric, threshold))

    def __iter__(self):
        return iter(self.elements)

    def __len__(self):
        return len(self.elements)


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

        # Assign dict elements to Parameters object and flatten 
        # thresholds
        self.__dict__.update(params)
        self.figsize = tuple(self.figsize) 
        self.thresholds_iterator = ThresholdIterator(self.thresholds)

        if self.audition_output_path is not None: 
            try:
                    with open(self.audition_output_path) as f:
                        json_models = json.load(f)

                    list_models = [model for model_list in \
                                   json_models.values() for \
                                   model in model_list]
                    self.model_group_id = list_models

            except FileNotFoundError:
                warnings.warn(
                    f'''No audition output file: 
                    {self.audition_output_path} 
                    was founded. Please check your Audition file PATH. 
                    I will use the models defined in the {path_params}
                    configuration file.''') 
