import pandas as pd
import numpy as np
import logging
import multiprocessing as mp

from typing import Dict
from multiprocessing import Process

from src.triage.component.ducktales.utils import distribute_jobs


class ExplainerBase:
    def __init__(self, model_object):
        """ Partially implemented abtract super class of the local explanations mehotds

            Args:
                model_object (int): model object to be explained
        """

        self.model_object = model_object
        
    def explain_example(self, example: np.array, pred_class=1) -> Dict[str, float]:
        """explain an individal instance. Subclasses should implement this"""
        
        raise NotImplementedError

    def explain_dataset(self, dataset, results_list, pred_class=1) -> pd.DataFrame:
        """explain a given dataset. Subclass should implement this

            Args:
                dataset (pd.DataFrame): The collection of records to be explained
                results_list: 
                    The mutiprocessing List type object that is shared across processes. 
                    The results of the sub processes are shared to the main process through this List. 
                pred_class:
                    The class label for which the explanation is given  
        
        """
        raise NotImplementedError


    def get_individual_feature_importance_scores(self, dataset, pred_class=1, n_jobs=1): 
        """ Implements the relevant explain_dataset function from the subclass and parallelizes the explanation process
        
            Args:
                dataset (pd.DataFrame): The instances
                pred_class (int): 
                    The class label for which the explanation is given. 
                    In triage applications, typically the classification is binary and the explanation defaults to the positive class
                n_jobs (int): 
                    Number of processes to be used for explaining the datapoints. 
                    Data parallelism used. Defaults to single core. If -1, all available cores are used
        """

        manager = mp.Manager()
        results_list = manager.list()

        n_records = len(dataset)
        if n_jobs == 1:
            logging.info('Explaining {} records using a single core'.format(n_records))

            return self.explain_dataset(dataset=dataset, results_list=results_list, pred_class=pred_class)

        n_cpu = mp.cpu_count()

        if (n_jobs > n_cpu) or (n_jobs == -1):
            n_jobs = n_cpu

        job_chunks = distribute_jobs(n_elements=n_records, n_jobs=n_jobs)

        logging.info('Explaining {} records parallelized across {} cores'.format(n_records, n_jobs))
        logging.info('Jobs distribution -- {}'.format(job_chunks))

        jobs = list()
        idx_cursor = 0
        for i, chunk_size in enumerate(job_chunks):
            p = Process(
                name='p{}'.format(i),
                target=self.explain_dataset,
                kwargs={
                    'dataset': dataset.iloc[idx_cursor: (idx_cursor+chunk_size)].values,
                    'pred_class': pred_class
                }
            )

            jobs.append(p)
            p.start()

            idx_cursor = idx_cursor + chunk_size

        for p in jobs:
            p.join()

        feature_importance_mat = results_list[0].append(results_list[1:])

        return feature_importance_mat













