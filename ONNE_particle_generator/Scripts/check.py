from glob import glob
import concurrent.futures as confu
import numpy as np
import json
import os
from tqdm import tqdm
import argparse
from concurrent import futures


#args = parser.parse_args()
#ncores=args.cores

def check_files(filename,i):
    with open(filename[i]) as f:
        data = json.load(f)
    try:    
        k=np.array(data['k'],dtype=float)
        chi=np.array(data['chi'],dtype=float)
    except:
        os.remove(filename[i])
def run(tasks,workpath):
    readout=glob(f"{workpath}/output/*.json")
    print(f"origin:{len(readout)}.")
    if type(readout)==str:
        readout=[readout]
    delete=0
    with tqdm(total=len(readout)) as pbar:
        with confu.ThreadPoolExecutor(max_workers=tasks) as executor:
            jobs=[executor.submit(check_files,readout,i) for i in range(len(readout))]
            for job in futures.as_completed(jobs):
                pbar.update(1)

    read_after=glob(f"{workpath}/output/*.json")
    print(f"remain: {len(read_after)}.")
