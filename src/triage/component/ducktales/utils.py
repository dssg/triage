import boto3
import yaml


def destribute_jobs(n_elements, n_jobs):
    """Get the chunk sizes for distributing jobs across multiple cores. This is intended for data parallelism

        args:
            n_elements (int): Number of elements to be ditributed across the cores
            n_jobs (int): Number of jobs/cores
    """

    chunk_size, remainder = divmod(n_elements, n_jobs)

    chunks_list = [chunk_size] * n_jobs

    # if there exists a remainder, we distribute them one-by-one
    for i in range(remainder):
        chunks_list[i] = chunks_list[i] + 1 

    return chunks_list





    