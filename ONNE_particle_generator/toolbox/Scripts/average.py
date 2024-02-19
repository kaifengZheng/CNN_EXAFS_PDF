import re
from scipy.interpolate import interp1d
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import concurrent.futures as confu
from concurrent import futures
from tqdm import tqdm


# parser=argparse.ArgumentParser(description="calculate average spectrum,defualt is using 1 core")
# parser.add_argument('-n','--cores',type=int,default=1,help='the number of works')

#args = parser.parse_args()
#ncores=args.cores
def load_files(filename,i):
    particle=filename[i].split('/')[-1].split("_site")[0]
    #print(particle)
    with open(filename[i]) as f:
        data = json.load(f)
    k=np.array(data['k'],dtype=float)
    chi=np.array(data["chi"],dtype=float)
    site=int(filename[i].split("_site_")[1].split('_')[0])
    n_sites=int(filename[i].split('_n_')[1].split('.')[0])
    return particle,k,chi,site,n_sites



def run(tasks,con_num,workpath):
    print("reading data...\n")
    output=dict()
    k_list=[]
    filename=glob.glob(f"{workpath}/output/*.json")
    with tqdm(total=len(filename)) as pbar:
        with confu.ProcessPoolExecutor(max_workers=tasks) as executor:
            jobs=[executor.submit(load_files,filename,i) for i in range(len(filename))]
            for job in futures.as_completed(jobs):
                k_list.append(job.result()[1])
                output[job.result()[0]]={'k':job.result()[1],
                            'chi':job.result()[2],
                            'site':job.result()[3],
                            'n_sites':job.result()[4]}
                pbar.update(1)



    keys=list(output.keys())
    #print(keys)
    print("\nremeshing energy grids.../n")
    k=np.array(k_list,dtype=float)
    #print(k)
    #mink,maxk=np.max(k[:,0]),np.min(k[:,-1])
    grids=np.linspace(0,10,1000)
    #print(output.keys())
    for key in tqdm(output.keys()):
        output[key]['chi']=interp1d(output[key]['k'],output[key]['chi'])(grids)
        output[key]['k']=grids
        #print(output[key])
    spectrum=dict()
    spectrum["k"]=grids
    for key in  output.keys():
        spectrum[key]=np.array(output[key]['chi'])
    #print(list(output.keys()))
    print("\naveraging spectra")
    con_sp=dict()
    con_sp['k']=np.array(spectrum['k'])
   
    config1=[keys[g] for g in range(len(keys)) if keys[g].__contains__(f'config_{con_num}_')]
    #print(config1)
    temp=np.zeros(len(spectrum['k']))
    for j in range(len(config1)):
        temp+=spectrum['k']**2*spectrum[config1[j]]
    con_sp[f'config_{con_num}']=temp/len(config1)
    print(f"len={len(config1)}")
    return con_sp





    # print("plotting...\n")

    # pd.DataFrame(con_sp).to_csv("../output_ave_test_kx.csv",index=False)

    # data=pd.read_csv("../output_ave_test_kx.csv")
    # fig,ax=plt.subplots()
    # for key in data.keys():
    #     if key!="k":
    #         #ax.plot(data['k'],data["k"]**2*data[key])
    #         ax.plot(data['k'],data[key])
    # ax.set_xlabel("k")
    # ax.set_ylabel("$k^2\chi$")
    # plt.title("average spectra")

    # #plt.show()
    # plt.savefig("../average_test_kx.png")
