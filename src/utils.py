import time
import logging
import sys

logging.basicConfig(level=logging.INFO, stream=sys.stdout)


def time_it(func):
    def measure_time(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        duration = end_time - start_time
        logging.info(f"Executing {func.__name__} took {duration:.5f} seconds.")
        return result
    return measure_time
