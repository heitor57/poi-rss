import time
from concurrent.futures import ProcessPoolExecutor
import sys
from multiprocessing.reduction import ForkingPickler, AbstractReducer
import multiprocessing as mp

from tqdm import tqdm
#from pathos.multiprocessing import ProcessingPool as Pool

class ForkingPickler4(ForkingPickler):
    def __init__(self, *args):
        if len(args) > 1:
            args[1] = 2
        else:
            args.append(2)
        super().__init__(*args)

    @classmethod
    def dumps(cls, obj, protocol=4):
        return ForkingPickler.dumps(obj, protocol)


def dump(obj, file, protocol=4):
    ForkingPickler4(file, protocol).dump(obj)


class Pickle4Reducer(AbstractReducer):
    ForkingPickler = ForkingPickler4
    register = ForkingPickler4.register
    dump = dump

def run_parallel(func, args,chunksize = 10):

    ctx = mp.get_context()
    ctx.reducer = Pickle4Reducer()

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
