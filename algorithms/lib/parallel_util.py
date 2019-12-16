import time
from concurrent.futures import ProcessPoolExecutor
import sys

from tqdm import tqdm
#from pathos.multiprocessing import ProcessingPool as Pool

def run_parallel(func, args,chunksize = 10):
    executor = ProcessPoolExecutor()
    num_args = len(args)
    results = [i for i in tqdm(executor.map(func,*list(zip(*args)),chunksize=chunksize),total=num_args)]
    # pool = Pool()
    # num_args = len(args)
    # results = pool.amap(func,*list(zip(*args)),chunksize=chunksize)
    # tmax = results._number_left
    # with tqdm(total=tmax) as pbar:
    #     while not results.ready():
    #         pbar.n = tmax - results._number_left
    #         pbar.refresh()
    #         time.sleep(1)
    # results = results.get()
    return results


# results = [i for i in tqdm(pool.map(func,*list(zip(*args)),chunksize=chunksize),total=num_args)]



# def run_parallel_rr(obj, func, args, chunksize = 10):
#     input("waiting enter")
#     with multiprocessing.Manager() as manager:
#         executor = ProcessPoolExecutor()
#         for i in range(len(args)):
#             args[i] = (obj,) + args[i]
#         num_args = len(args)
#         results = [i for i in tqdm(executor.map(func,*list(zip(*args)),chunksize=chunksize),total=num_args)]
#         return results
