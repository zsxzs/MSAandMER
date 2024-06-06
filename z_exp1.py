# add ple 
import os
import sys
sys.path.append(os.path.dirname(__file__))
from run import MMSA_run


if __name__ == '__main__':

    # run LMF on MOSI with default hyper parameters
    # MMSA_run('lmf', 'mosi', seeds=[1111, 1112, 1113], gpu_ids=[0])

    # tune Self_mm on MOSEI with default hyper parameter range
    # MMSA_run('self_mm', 'mosei', seeds=[], gpu_ids=[0])
    MMSA_run('self_mm', 'mosei', seeds=[], gpu_ids=[0])