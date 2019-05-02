import random


FLOAT_TO_INT_MULTIPLIER = 2000000000


def generate_python_random_seed():
    """Generate a random integer suitable for seeding the Python random generator
    """
    return int(random.uniform(0, 1.0) * FLOAT_TO_INT_MULTIPLIER)


def convert_python_random_seed_to_postgres(python_random_seed):
    return python_random_seed / FLOAT_TO_INT_MULTIPLIER
