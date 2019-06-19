from __future__ import print_function
from functools import wraps, partial
import time

def timeit_if(func, threshold):

    @wraps(func)
    def time_fun(*args, **kwargs):
        if not threshold: 
            res = func(*args, **kwargs)
        elif threshold > 1 :
            start_time = time.time()
            print("--- profile ", func.__name__, "---")
            res = func(*args, **kwargs)
            print("--- %s seconds ---" % (time.time() - start_time))
        elif threshold > 0 :
            print("--- run ", func.__name__, "---")
            res = func(*args, **kwargs)
            print("--- done  ---")
        return res

    return time_fun
