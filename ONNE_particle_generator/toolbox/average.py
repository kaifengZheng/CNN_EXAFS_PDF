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
import toml


parser=argparse.ArgumentParser(description="calculate average spectrum,defualt is using 1 core")
parser.add_argument('-n','--cores',type=int,default=1,help='the number of works')

args = parser.parse_args()
ncores=args.cores

config=toml.load('../config.toml')


def load_files(filename,i):
    particle=filename[i].split('/')[2].split(".")[0]
    with open(filename[i]) as f:
      data = json.load(f)
    k=np.array(data['k'],dtype=float)
    chi=np.array(data["chi"],dtype=float)
    #print(particle)
    #print(chi)
    site=int(filename[i].split("_site_")[1].split('_')[0])
    n_sites=int(filename[i].split('n_')[1].split('.')[0])
    return particle,k,chi,site,n_sites




print("reading data...\n")
output=dict()
k_list=[]
filename=glob.glob("../output/*.json")
#print(filename)
with tqdm(total=len(filename)) as pbar:
    with confu.ProcessPoolExecutor(max_workers=ncores) as executor:
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
mink,maxk=np.max(k[:,0]),np.min(k[:,-1])
grids=np.linspace(mink,maxk,200)
for key in tqdm(output.keys()):
    output[key]['chi']=interp1d(output[key]['k'],output[key]['chi'])(grids)
    output[key]['k']=grids
#print(output)
spectrum=dict()
spectrum["k"]=grids
for key in  output.keys():
    spectrum[key]=np.array(output[key]['chi'])

print("\naveraging spectra")
con_sp=dict()
diff_k=[]

temp=np.zeros(len(spectrum['k']))
con_sp['k']=np.array(spectrum['k'])
#for i in tqdm(range(1,len(keys)),total=len(keys)-1):
    #config1=[keys[g] for g in range(len(keys)) if keys[g].__contains__(diff_k[i])]
    #print(config1)
    #temp=np.zeros(len(spectrum['k']))
    #temp+=spectrum['k']**2*spectrum[keys[i]]
#    temp+=spectrum[keys[i]]
#con_sp['UF4']=temp/len(keys)
#print(f'Total number of sites:{len(keys)}\n')
con_sp['k']=grids
#print("plotting...\n")
for i in tqdm(range(config['dataset']['conf_num']),total=config['dataset']['conf_num']):
    config1=[keys[g] for g in range(len(keys)) if keys[g].__contains__(f'config_{i+1}_')]
    #print(config1)
    temp=np.zeros(len(spectrum['k']))
    if len(config1)<80:
        continue
    for j in range(len(config1)):
        temp+=spectrum[config1[j]]
    con_sp[f'config_{i+1}']=temp/len(config1)


pd.DataFrame(con_sp).to_csv("../output_ave_test_kx.csv",index=False)

data=pd.read_csv("../output_ave_test_kx.csv")
fig,ax=plt.subplots()
for key in data.keys():
    if key!="k":
        #ax.plot(data['k'],data["k"]**2*data[key])
        ax.plot(data['k'],data[key])
ax.set_xlabel("k")
ax.set_ylabel("$\chi$")
plt.title("average spectra")

#plt.show()
plt.savefig("../average_test_kx.png")
