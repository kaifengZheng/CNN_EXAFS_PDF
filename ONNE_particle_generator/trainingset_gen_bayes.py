import sys 
import pickle
import os
import toml
import json
import matplotlib.pyplot as plt 
import subprocess
import shutil
import csv
config=toml.load('config.toml')
#print(config['dataset']['frames'])
sys.path.append(os.path.abspath(config['file']['scriptpath']))

from FEFF_run_v2 import *
from EXAFS_sim import *
import average
import check
import scipy.stats as stats
import numpy as np
from scipy.optimize import dual_annealing
from scipy.interpolate import interp1d
import pickle
import toml
from tqdm import tqdm
import pandas as pd
import concurrent.futures as confu
from concurrent import futures
from functools import partial
import GPyOpt
from GPyOpt.experiment_design import initial_design
from glob import glob
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw


# mesh of points between 0 and 2pi
theta_mesh = np.linspace(0, np.pi, 200)
phi_mesh = np.linspace(0, 2*np.pi, 200)

# lower, upper = 2, 3
# mu, sigma = 2.5, 0.1
# S1 = stats.norm(loc=mu, scale=sigma)

# lower, upper = 3.2, 4
# mu, sigma = 3.5, 0.1
# S2 = stats.norm(loc=mu, scale=sigma)

# lower, upper = 4.2, 6
# mu, sigma = 5, 0.1
# S3 = stats.norm(loc=mu, scale=sigma)



def save_p(data, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
def open_p(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)


def distro_distance(S):
    return S.rvs(1)

def from_xyz(xyz, axis=-1):
    x, y, z = np.moveaxis(xyz, axis, 0)

    lea = np.empty_like(xyz)

    pre_selector = ((slice(None),) * lea.ndim)[:axis]

    xy_sq = x ** 2 + y ** 2
    lea[(*pre_selector, 0)] = np.sqrt(xy_sq + z ** 2)
    lea[(*pre_selector, 1)] = np.arctan2(np.sqrt(xy_sq), z)
    lea[(*pre_selector, 2)] = np.arctan2(y, x)

    return lea


def to_xyz(lea, axis=-1):
    l, e, a = np.moveaxis(lea, axis, 0)

    xyz = np.empty_like(lea)

    pre_selector = ((slice(None),) * xyz.ndim)[:axis]

    xyz[(*pre_selector, 0)] = l * np.sin(e) * np.cos(a)
    xyz[(*pre_selector, 1)] = l * np.sin(e) * np.sin(a)
    xyz[(*pre_selector, 2)] = l * np.cos(e)

    return xyz


# define a function to get a random float between 0 and np.pi
def get_random_theta():
    return np.random.uniform(0,np.pi)

# define a function to get a random point between 0 and 360
def get_random_phi():
    return np.random.uniform(0,2*np.pi)

def get_random_r():
    return np.random.uniform(1,6)

def d_sphere(rtp1, rtp2):
    r1, th1, ph1 = rtp1
    r2, th2, ph2 = rtp2
    return np.sqrt(r1**2+r2**2-2*r1*r2*np.cos((th1-th2))-2*r1*r2*np.sin(th1)*np.sin(th2)*(np.cos((ph1-ph2))-1))

def sph_dis_matrix(list_of_points):
    dm = np.zeros((len(list_of_points),len(list_of_points)))
    for i in range(len(list_of_points)):
        for j in range(len(list_of_points)):
            dm[i,j]=d_sphere(list_of_points[i], list_of_points[j])
    return dm

def d_util_da(v):
    global good_points, r_target, d_target
    global log
    th2, ph2 = v
    if len(good_points)>2:
        check_points = np.concatenate((good_points, np.array([[r_target, th2, ph2]])))
        d0 = find_min(check_points)
    else:
        r1 = good_points[-1][0]
        th1 = good_points[-1][1]
        ph1 = good_points[-1][2]
        r2 = r_target
        d0=np.sqrt(r1**2+r2**2-2*r1*r2*np.cos((th1-th2))-2*r1*r2*np.sin(th1)*np.sin(th2)*(np.cos((ph1-ph2))-1))
        

    if d0 < d_target:
        return d0+d_target
    else:
        return d0

def find_min(list_of_points):
    tri = sph_dis_matrix(list_of_points)
    #tri = sph_dis_matrix(list_of_points[1:])
    tri_no_zeros = tri[tri!=0]
    
    return min(tri_no_zeros)

def find_a_point(radial, dis):
    global good_points, r_target, d_target
    r_target = radial
    d_target = dis
    result = dual_annealing(d_util_da, bounds)
    th2, ph2 = result.x
    return np.array([r_target, th2, ph2])

bounds = [[0, np.pi], [0, 2*np.pi]]
def get_rdf_abs(coords, abs_el1, rmesh):
    dm = fastdist.matrix_pairwise_distance(coords, fastdist.euclidean, "euclidean", return_matrix=True)
    mean = np.mean(coords, axis=0)
    all_ = fastdist.vector_to_matrix_distance(mean, coords, fastdist.euclidean, "euclidean")
    abs_el1 = dm[abs_el1]
    counts_abs_el1 = bin_list_mono(abs_el1, rmesh)
    return np.asarray(counts_abs_el1)
def get_rdf_abs(coords, abs_el1, rmesh):
    dm = fastdist.matrix_pairwise_distance(coords, fastdist.euclidean, "euclidean", return_matrix=True)
    mean = np.mean(coords, axis=0)
    all_ = fastdist.vector_to_matrix_distance(mean, coords, fastdist.euclidean, "euclidean")
    abs_el1 = dm[abs_el1]
    counts_abs_el1 = bin_list_mono(abs_el1, rmesh)
    return np.asarray(counts_abs_el1)


def normal_distro_S1(mean, std):
    global S1
    lower, upper = mean-.5, mean+.5
    mu, sigma = mean, std
    S1 = stats.norm(loc=mu, scale=sigma)

def normal_distro_S2(mean, std):
    global S2
    lower, upper = mean-.5, mean+.5
    mu, sigma = mean, std
    S2 = stats.norm(loc=mu, scale=sigma)

def normal_distro_S2d(mean, std):
    global S2d
    lower, upper = mean-.5, mean+.5
    mu, sigma = mean, std
    S2d = stats.norm(loc=mu, scale=sigma)

def normal_distro_S3(mean, std):
    global S3
    lower, upper = 4,6
    mu, sigma = mean, std
    S3 = stats.norm(loc=mu, scale=sigma)

def normal_distro_S4(mean, std):
    global S4
    lower, upper = mean-.5, mean+.5
    mu, sigma = mean, std
    S4 = stats.norm(loc=mu, scale=sigma)

def d_util_da_s(v):#important


    global bond_matrix
    global operate




    n_rows = len(bond_matrix)
    all_rows = np.arange(n_rows)
    operate_rows = operate
    other_rows = np.setdiff1d(all_rows, operate_rows)

    n_operate = len(operate_rows)
    random_t_p_matrix = np.zeros((n_operate,2))

    for i in range(0,n_operate):
        random_t_p_matrix[i,0]=v[i]
    for i in range(n_operate, 2*n_operate):
        random_t_p_matrix[i-n_operate,1]=v[i]


    min_matrix=np.zeros((n_rows,3))

    for i in all_rows:
        min_matrix[i,0]=bond_matrix[i,0]

    for i,j in enumerate(operate_rows):
        min_matrix[j,1]=random_t_p_matrix[i,0]
        min_matrix[j,2]=random_t_p_matrix[i,1]

    for i in other_rows:
        min_matrix[i,1]=bond_matrix[i,1]
        min_matrix[i,2]=bond_matrix[i,2]
    
    min_val = find_min(min_matrix)
    x=min_val
    target_min = 3 #important minimum distance
    var=4
    return .2+-np.exp(-np.square(x-target_min)/2*var)/(np.sqrt(2*np.pi*var))
        
    
def matrix_reform(v):

    global bond_matrix
    global operate

    n_rows = len(bond_matrix)
    all_rows = np.arange(n_rows)
    operate_rows = operate
    other_rows = np.setdiff1d(all_rows, operate_rows)

    n_operate = len(operate_rows)

    b = v[:n_operate]
    c = v[n_operate:n_operate*2]
    random_t_p_matrix = np.array([b,c]).transpose()


    min_matrix=np.zeros((n_rows,3))

    for i in all_rows:
        min_matrix[i,0]=bond_matrix[i,0]

    for i,j in enumerate(operate_rows):
        min_matrix[j,1]=random_t_p_matrix[i,0]
        min_matrix[j,2]=random_t_p_matrix[i,1]

    for i in other_rows:
        min_matrix[i,1]=bond_matrix[i,1]
        min_matrix[i,2]=bond_matrix[i,2]


    
    return min_matrix
    

def make_rdf_feff(distances):
    global rmeshPrime
    digitized =np.digitize(distances
        , rmeshPrime)
    #min_d=min(distances)
    #max_d=max(distances)
    unique, counts = np.unique(digitized, return_counts=True)
    counter = [0]*len(rmeshPrime)
    for i in range(len(unique)):
        counter[unique[i]-1] = counts[i]/(rmeshPrime[1]-rmeshPrime[0])
    return np.asarray(counter)

def make_xyz(stuff):
    str_text=""
    for i in range(len(stuff[0])):
        str_text+=stuff[0][i]+" "+str(stuff[1][i][0])+" "+str(stuff[1][i][1])+" "+str(stuff[1][i][2])+"\n"
    return str_text
def optimization(partial_bonds,n_atoms_total,frame):
    global bond_matrix
    global operate
    #print(n_atoms_total)
    bond_intervals = [len(l) for l in partial_bonds[frame]]
    #print(f"bond_intervals={bond_intervals}\n")
    bond_index = list(range(n_atoms_total))
    bond_index = np.split(bond_index, np.cumsum(bond_intervals)[:-1])
    #print(f"bond_index={bond_index}")
    #print(f"bond_index={bond_index}")
    for i in range(len(partial_bonds[frame])):
        #print(f'frame={len(partial_bonds[frame])}')
        rhos = partial_bonds[frame][i]
        n_rows = len(rhos)
        bond_matrix=np.zeros((n_rows,3))
        bond_matrix_c1=rhos
        bond_matrix[:,0]=bond_matrix_c1

        if i==0:

            operate = bond_index[i]

            bounds = np.concatenate((np.array([[0, np.pi] for i in range(len(operate))]), np.array([[0, 2*np.pi] for i in range(len(operate))])))
            #print(bounds)
            result = dual_annealing(d_util_da_s, bounds, maxiter=10, initial_temp=5230, visit=2, accept=-5, seed=1234)
            result = result["x"]

            saved = matrix_reform(result)
        
        else:
            bond_matrix = np.concatenate((saved, bond_matrix))

            operate = bond_index[i]
        

            bounds = np.concatenate((np.array([[0, np.pi] for i in range(len(operate))]), np.array([[0, 2*np.pi] for i in range(len(operate))])))
            
            result = dual_annealing(d_util_da_s, bounds, maxiter=10, initial_temp=5230, visit=2, accept=-5, seed=1234)

            result = result["x"]
            
            saved = matrix_reform(result)
        #print(f"bounds={bounds}\n")
        #print(f"operate={operate}\n")
        #print(f"saved={saved}\n")
    #print(saved)
    return saved

def atom_number(layer_symbol,layer_num,layer_per):
    num=len(layer_symbol)
    s_num=[]
    for i in range(num-1):
        s_num.append(np.ceil(layer_per[i]*layer_num))
    s_num.append(layer_num-np.sum(s_num))
    return s_num
def create_frames(con, con_num,config):
    """
    @param con: selected parameters for optimization
    @param config: system configurations in config.toml
    @param con_num: configuration number
    """
    global rmeshPrime

    config_layer1_s=config['structure']['layer1_symbol']#a list fixed
    config_layer2_s=config['structure']['layer2_symbol']#a list fixed
    config_layer3_s=config['structure']['layer3_symbol']# a list fixed
    config_layer4_s=config['structure']['layer4_symbol']# a list fixed
    normal_distro_S1(con[0], con[1])
    normal_distro_S2(con[2], con[3])
    normal_distro_S3(con[4], con[5])
    normal_distro_S4(con[6], con[7])
    
    n_frames = config["dataset"]["frames"]
    # n_atoms_outer=11
    # n_cl=con[1]
    # n_zn=2
    # n_frac=.5
    # n_cl_e=round(n_frac*n_atoms_outer)
    # n_k=n_atoms_outer-n_cl_e
    layer2_s_num=atom_number(config_layer2_s,con[9],[con[12]])
    layer3_s_num=atom_number(config_layer3_s,con[10],[con[13]])
    layer4_s_num=atom_number(config_layer4_s,con[11],[con[14]])
    #print(con[8])
    #print(config_layer1_s)
    for i in range(len(config_layer2_s)):
        if layer2_s_num[i]==0:
           # del layer2_s_num[i]
           layer2_s_num.pop(i)
           config_layer2_s.pop(i)
    #print(layer2_s_num)
    #print(config_layer2_s)
    
    for i in range(len(config_layer3_s)):
        if layer3_s_num[i]==0:
           # del layer2_s_num[i]
           layer3_s_num.pop(i)
           config_layer3_s.pop(i)
    #print(layer3_s_num)
    #print(config_layer3_s)

    for i in range(len(config_layer4_s)):
        if layer4_s_num[i]==0:
           # del layer2_s_num[i]
           layer4_s_num.pop(i)
           config_layer4_s.pop(i)
    #print(layer4_s_num)
    #print(config_layer4_s)
    sum_atom=np.int32(np.sum([con[8]])+np.sum(layer2_s_num)+np.sum(layer3_s_num)+np.sum(layer4_s_num))


    partial_bonds = []
    #sum_atom=0
    for i in range(n_frames):
        frame_list = []
        frame_list.append(S1.rvs(int(con[8])))
        for j in range(len(config_layer2_s)):
            #sum_atom+=int(layer2_s_num[j])
            frame_list.append(S2.rvs(int(layer2_s_num[j])))

        for j in range(len(config_layer3_s)):
            #sum_atom+=int(layer3_s_num[j])
            frame_list.append(S3.rvs(int(layer3_s_num[j])))

        for j in range(len(config_layer4_s)):
            #sum_atom+=int(layer4_s_num[j])
            frame_list.append(S4.rvs(int(layer4_s_num[j])))
        partial_bonds.append(frame_list)
        #print(f"sum_atom={sum_atom}")
        #print(f'frame_list={frame_list}')
        #print(partial_bonds)
    #print(len(partial_bonds))
    #all_bonds = np.concatenate((np.array([l[0] for l in partial_bonds]).flatten(),np.array([l[1] for l in partial_bonds]).flatten(),np.array([l[2] for l in partial_bonds]).flatten(),np.array([l[3] for l in partial_bonds]).flatten()))

    #plt.hist(all_bonds, bins=100)
    #plt.show()
    rdf=[]
    rdf1=[]
    for i in range(len(partial_bonds)):
        temp=[]
        for j in range(len(partial_bonds[i][0])):
            temp.append(partial_bonds[i][0][j])
        rdf1.append(make_rdf_feff(temp))
        for j in range(len(partial_bonds[i][1])):
            temp.append(partial_bonds[i][1][j])
        for j in range(len(partial_bonds[i][2])):
            temp.append(partial_bonds[i][2][j])
        for j in range(len(partial_bonds[i][3])):
            temp.append(partial_bonds[i][3][j])
        rdf.append(make_rdf_feff(temp))

    #print(save_con)
    n_atoms_total=sum_atom
    #print(f"n_atoms={n_atoms_total}")



    optimized_results = []
    
    optimize_func=partial(optimization,partial_bonds,n_atoms_total)
    #print(partial_bonds)
    #try:
        #for frame in tqdm(range(len(partial_bonds)),total=len(partial_bonds)):
    with confu.ProcessPoolExecutor(max_workers=tasks) as executor:
        jobs=[executor.submit(optimize_func,frame) for frame in range(len(partial_bonds))]
        for job in futures.as_completed(jobs):
            saved=job.result()
            optimized_results.append(saved)
    #except ValueError:
    #except Exception as e:
    #    pass
        #print("bad structure!")
        #with open("output.dat",'a') as file1:
        #    file1.write(f"{e}\n")
            #save_con.pop(con_num)


    rdf=np.array(rdf).mean(axis=0)
    rdf1=np.array(rdf1).mean(axis=0)
    
    # #plot
    #ax.plot(rmeshPrime,rdf,color=f"C{con_num}",label=f"config {con_num}")

    
    
    save_gr=rdf
    save_gr1=rdf1





    all_results = np.array(optimized_results)
    atom_ids=[]
    atom_symbol_dic_str_to_int=config['structure']['atom_dict']
    for i in range(len(config_layer1_s)):
        for j in range(int(con[8])):
            atom_ids.append([config_layer1_s[i]])
    
    for i in range(len(config_layer2_s)):
        for j in range(int(layer2_s_num[i])):
            atom_ids.append([config_layer2_s[i]])
    for i in range(len(config_layer3_s)):
        for j in range(int(layer3_s_num[i])):
            atom_ids.append([config_layer3_s[i]])
    for i in range(len(config_layer4_s)):
        for j in range(int(layer4_s_num[i])):
            atom_ids.append([config_layer4_s[i]])


    atom_ids=np.asarray(atom_ids).astype('<U32').reshape(-1,1)
    #print(atom_ids)
    atom_ids_int = [atom_symbol_dic_str_to_int[l] for l in np.insert(atom_ids,0,[config['structure']["center"]]).flatten()]
    #print(atom_ids)
    final_points_xyz = list(map(lambda x: to_xyz(x), all_results))
    final_points_xyz = list(map(lambda x: np.insert(x,0,np.array([0,0,0]),axis=0), final_points_xyz))
    master_set_x = list(map(lambda x: np.insert(x, 0, atom_ids_int, axis=1), final_points_xyz))
    master_set_x = np.array(master_set_x)

    #ms_header_0=config['feff']['header']

    #ms_header=make_head(ms_header_0)
    atom_symbol_dic=dict()
    ele=list(atom_symbol_dic_str_to_int.keys())
    num=list(atom_symbol_dic_str_to_int.values())
    for i in range(len(atom_symbol_dic_str_to_int)):
        atom_symbol_dic[str(num[i])]=ele[i]
    main=config['file']['workpath']

    #example=example
    try:
        os.makedirs(f'input{con_num}')
    except:
        pass

    for i, stuff in enumerate(master_set_x):

        out=make_xyz([[atom_symbol_dic[str(int(s))] for s in stuff[:,0]],stuff[:,1:]])
    
        with open(f"input{con_num}/config_{con_num}_frame_{i}_site_0.xyz", "w") as f:
            f.write(f"{len(stuff)}\n")
            f.write(f"UF4_test\n")
            f.write(out)
    save_con={f"config.{con_num}":{"layer1_mean":str(con[0]),
                      "layer1_std":str(con[1]),
                      "layer2_mean":str(con[2]),
                      "layer2_std":str(con[3]),
                      "layer3_mean":str(con[4]),
                      "layer3_std":str(con[5]),
                      "layer4_mean":str(con[6]),
                      "layer4_std":str(con[7]),
                      "layer1_comp":config_layer1_s,
                      "layer1_comp_num":str(np.int32([con[8]])),
                      "layer2_comp":config_layer2_s,
                      "layer2_comp_num":str(np.int32(layer2_s_num)),
                      "layer3_comp":config_layer3_s,
                      "layer3_comp_num":str(np.int32(layer3_s_num)),
                      "layer4_comp":config_layer4_s,
                        "layer4_comp_num":str(np.int32(layer4_s_num))}
                      }
    #print(f"save_con={save_con}")
    with open("output.dat",'a') as file1:
        file1.write(f"configuration.{con_num}:{save_con}\n")

    return save_con,save_gr,save_gr1

def run_feff(con_num):
    if os.path.exists("input_FEFF"):
        shutil.rmtree("input_FEFF")
    if os.path.exists("output"):
        shutil.rmtree("output")
    if os.path.exists("input"):
        shutil.rmtree("input")
    os.mkdir("input")
    files=glob(f"input{con_num}/*.xyz")
    for s in files:
        shutil.copy2(s,"input")
    print("write....")
    subprocess.run(['python run_FEFF.py -w','wait'],shell=True)
    print("run...")
    subprocess.run(['python run_FEFF.py -r','wait'],shell=True)

def check_feff(tasks,workpath):
    print("checking...")
    check.run(tasks,workpath)
    
def average_spectra(tasks,con_num,workpath):
    print("calculating averaging...")
    return average.run(tasks,con_num,workpath)

#loss
def RRMSE(exp,cal):
    num = np.sum(np.square(exp - cal))
    den = np.sum(np.square(cal))
    squared_error = num/den
    rrmse_loss = np.sqrt(squared_error)
    return rrmse_loss
def mse(exp,cal):
    return np.sqrt(np.sum(np.square(exp - cal)))
def RRMSE_MSE(exp,cal):
    return RRMSE(exp,cal)*mse(exp,cal)
def correlation(exp, cal):
    a_diff = exp - np.mean(exp)
    p_diff = cal - np.mean(cal)
    numerator = np.sum(a_diff * p_diff)
    denominator = np.sqrt(np.sum(a_diff ** 2)) * np.sqrt(np.sum(p_diff ** 2)) 
    return -numerator / denominator
def fdtw_method(exp,cal):
    x=[exp]
    y=[cal]
    distance, path = fastdtw(x,y, dist=euclidean)
    return distance


def objective(configuration,config,con_num):


        
    k_grids=np.linspace(0,10,1000)

    data=np.loadtxt(config['exp']['path'])
    x2_chi=data[:,3]
    k_exp=data[:,0]
    exp_chi=interp1d(k_exp,x2_chi,kind='cubic')(k_grids)


    
    con_save,gr,gr1=create_frames(configuration, con_num,config)

    run_feff(con_num)
    check_feff(config['mpi']['tasks'],config['file']['workpath'])
    chi=average_spectra(config['mpi']['tasks'],con_num,config['file']['workpath'])
    with open("chi.csv",'a') as csvfile:
        filewriter=csv.writer(csvfile,delimiter=',',lineterminator='\n')
        filewriter.writerow(chi[f"config_{con_num}"])
    with open('gr.csv','a') as csvfile:
        filewriter=csv.writer(csvfile,delimiter=',',lineterminator='\n')
        filewriter.writerow(gr)
    with open('gr1.csv','a') as csvfile:
        filewriter=csv.writer(csvfile,delimiter=',',lineterminator='\n')
        filewriter.writerow(gr1)
    file=open('configuration.txt','a')
    json.dump(con_save,file)
    file.close()
    #if con_num==1:
    fig,ax=plt.subplots(1,2)

    ax[1].plot(k_grids,exp_chi,color='r',label='exp')

    ax[0].plot(rmeshPrime,gr,label=f'conf_{con_num}')
    ax[0].set_xlabel('r($\AA$)')
    ax[0].set_ylabel('g(r)')
    ax[0].legend(frameon=False)
    ax[1].plot(k_grids,chi[f'config_{con_num}'],label=f"conf_{con_num}")
    ax[1].set_xlabel('k($\AA^{-1}$)')
    ax[1].set_ylabel('$k^2\chi(k)$ ($\AA^{-2}$)')
    ax[1].legend(frameon=False)
    plt.tight_layout()
    plt.savefig(f"plot/gr_EXAFS_config.{con_num}.png")
    print('spec='+str(chi[f'config_{con_num}']))
    print(f"exp={exp_chi}")
    #loss_value=np.sum(np.abs(chi[f'config_{con_num}']-exp_chi))
    loss_value=correlation(exp_chi,chi[f'config_{con_num}'])
    #loss_value=fdtw_method(exp_chi,chi[f'config_{con_num}'])
    with open("loss.out",'a') as file2:
        file2.write(f'{con_num},{loss_value}')
    return loss_value
    

    

def main():
    global rmeshPrime


    if os.path.exists("chi.csv"):
        os.remove("chi.csv")
    if os.path.exists("gr.csv"):
        os.remove("gr.csv")
    if os.path.exists("gr2.csv"):
        os.remove("gr2.csv")
    if os.path.exists("configuration.txt"):
        os.remove("configuration.txt")
    if os.path.exists("loss.out"):
        os.remove("loss.out")
    if os.path.exists("plot"):
        shutil.rmtree("plot")
    os.mkdir('plot')

    config=toml.load('config.toml')
    #print(config['dataset']['frames'])

    rmeshPrime = np.arange(0,10,0.05)
    np.savetxt("rmesh.txt",rmeshPrime,delimiter=',')
    con_num=1
    bounds_conf = [{'name':'layer1_m','type':'continuous','domain':(2,3.5)},
            {'name':"layer1_std",'type':'continuous','domain':(0.1,0.5)},
            {'name':"layer2_m",'type':'continuous','domain':(2.5,4.5)},
            {'name':"layer2_std","type":"continuous","domain":(0.1,0.5)},
            {'name':'layer3_m','type':"continuous",'domain':(3,5.5)},
            {'name':'layer3_std','type':'continuous','domain':(0.1,0.5)},
            {'name':'layer4_m','type':'continuous','domain':(4,6.2)},
            {'name':'layer4_std','type':'continuous','domain':(0.1,0.5)},
            {'name':'layer1_num','type':'discrete','domain':(7,8,9,10)},
            {'name':'layer2_num','type':'discrete','domain':(4,5,6,7,8,9,10,11,12)},
            {'name':'layer3_num','type':'discrete','domain':(4,5,6,7,8,9,10,11,12)},
            {'name':'layer4_num','type':'discrete','domain':(4,5,6,7,8,9,10,11,12)},
            {'name':'layer2_F','type':'continuous','domain':(0.6,1)},
            {'name':'layer3_F','type':'continuous','domain':(0.6,1)},
            {'name':'layer4_F','type':'continuous','domain':(0.6,1)}]
    space=GPyOpt.Design_space(space=bounds_conf)
    X_init=initial_design('random',space,1)
    Y_init=objective(X_init[0],config,con_num)
    X_step=X_init
    Y_step=[[Y_init]]
    print(f"X={X_step}")
    print(f"Y={Y_step}")
    loss=[]
    n_con=[]
    if Y_init<1000:
        loss.append(Y_step[con_num-1][0])
        n_con.append(con_num)
        fig,ax=plt.subplots()
        ax.plot(n_con,loss,'r-.')
        ax.set_xlabel("configuration")
        ax.set_ylabel('loss')
        plt.tight_layout()
        plt.savefig(f"plot/loss{con_num}.png")


    con_num+=1
    while con_num<=config["dataset"]["conf_num"]:
        bo_step = GPyOpt.methods.BayesianOptimization(f = None, domain = bounds_conf,batch_size=10,num_cores=10 ,X = X_step, Y = Y_step)
        x_next = bo_step.suggest_next_locations()
        y_next = objective(x_next[0],config,con_num)
        X_step = np.vstack((X_step, x_next))
        Y_step = np.vstack((Y_step, y_next))
        print(f"loss={Y_step[con_num-1][0]}")
        #if Y_step[con_num-1][0]>10*np.max(loss):
        loss.append(Y_step[con_num-1][0])
        n_con.append(con_num)
        fig,ax=plt.subplots()
        ax.plot(n_con,loss,'ro-')
        ax.set_xlabel("configuration")
        ax.set_ylabel('loss')
        plt.tight_layout()
        plt.savefig(f"plot/loss{con_num}.png")
        test=np.where(np.abs(loss/np.min(loss))>500)[0]
        if len(test)!=0:
            loss.pop(test[0])
        con_num+=1
if __name__ == '__main__':
    main()
