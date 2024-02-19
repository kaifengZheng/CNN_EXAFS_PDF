#!/usr/bin/env python3
import sys 
import os
import toml
import matplotlib.pyplot as plt 

config=toml.load('config.toml')
sys.path.append(os.path.abspath(config['file']['scriptpath']))


from EXAFS_sim import *
import scipy.stats as stats
import numpy as np
from scipy.optimize import dual_annealing
import pickle
import toml
from tqdm import tqdm
import pandas as pd
import concurrent.futures as confu
from concurrent import futures
from functools import partial
import GPyOpt
from GPyOpt.experiment_design import initial_design
from scipy.stats import uniform

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
def uniform_background(scale):
    global S0
    S0 = uniform(scale=scale)

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
    dr=rmeshPrime[1]-rmeshPrime[0]
    distances=np.array(distances)
    distances=distances[distances<rmeshPrime[-1]]
    num_bin=[]
    V=4/3*np.pi*rmeshPrime[-1]**3
    N=len(distances)
    rho=N/V
    r=[]
    for i in range(len(rmeshPrime)-1):
        num=len(np.where((distances>=rmeshPrime[i]) & (distances<rmeshPrime[i+1]))[0])
        r_loc=np.round(rmeshPrime[i]+dr/2,6)
        num_nor=num/(4*np.pi*(r_loc**2)*dr) #delete the effect of rho
        r.append(r_loc)
        num_bin.append(np.round(num_nor,6))
    return r,num_bin,rho

def make_xyz(stuff):
    str_text=""
    for i in range(len(stuff[0])):
        str_text+=stuff[0][i]+" "+str(stuff[1][i][0])+" "+str(stuff[1][i][1])+" "+str(stuff[1][i][2])+"\n"
    return str_text
def optimization(partial_bonds,frame):
    global bond_matrix
    global operate
    #print(n_atoms_total)
    bond_intervals = [len(l) for l in partial_bonds[frame]]
    #print(f"bond_intervals={bond_intervals}")
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
        #print(partial_bonds)
        if i==0:

            operate = bond_index[i]

            bounds = np.concatenate((np.array([[0, np.pi] for i in range(len(operate))]), np.array([[0, 2*np.pi] for i in range(len(operate))])))
            #print(bounds)
            result = dual_annealing(d_util_da_s, bounds, maxiter=20, initial_temp=5230, visit=2, accept=-5, seed=1234)
            result = result["x"]

            saved = matrix_reform(result)
        
        else:
            bond_matrix = np.concatenate((saved, bond_matrix))

            operate = bond_index[i]
        

            bounds = np.concatenate((np.array([[0, np.pi] for i in range(len(operate))]), np.array([[0, 2*np.pi] for i in range(len(operate))])))
            #print(bounds) 
            result = dual_annealing(d_util_da_s, bounds, maxiter=20, initial_temp=5230, visit=2, accept=-5, seed=1234)

            result = result["x"]
            
            saved = matrix_reform(result)

        #print(f"bounds={bounds}\n")
        #print(f"operate={operate}\n")
        #print(f"saved={saved}\n")
    #print(f"{len(saved)}")
    return saved

def atom_number(layer_symbol,layer_num,layer_per):
    num=len(layer_symbol)
    print(f"l4={layer_per,layer_num}")
    s_num=[]
    F=np.ceil(layer_per[0]*layer_num)
    #residual=layer_num-F
    U=np.ceil(layer_per[1]*layer_num)
    Li=layer_num-F-U
    if Li<0:
        U=U+Li
        Li=0
    s_num=[F,U,Li]
    return s_num


rmeshPrime = np.arange(0.5,6,0.05)

# mean_layer1=config['structure']['mean1']#contains min max value, continuous2.3 3.5
# std_layer1=config['structure']['std1']#contains min max value, continuous
# mean_layer2=config['structure']['mean2']#contains min max value, continuous3.5 4.5
# std_layer2=config['structure']['std2']#contains min max value, continuous
# mean_layer3=config['structure']['mean3']#contains min max value, continuous4.5 5.5
# std_layer3=config['structure']['std3']#contains min max value, continuous
# mean_layer4=config['structure']["mean4"]#contains min max value, continuous5.5 6.5
# std_layer4=config['structure']['std4']#contains min max value, continuous
n_frames=config['dataset']['frames']




# configuration =[[a,b,c,d,e,f,g,h] for a in mean_layer1 
#                 for b in std_layer1 
#                 for c in mean_layer2 
#                 for d in std_layer2 
#                 for e in mean_layer3 
#                 for f in std_layer3
#                 for g in mean_layer4
#                 for h in std_layer4]
con_num=1
bounds_conf = [{'name':'layer1_m','type':'continuous','domain':(1.9,2.5)},
          {'name':"layer1_std",'type':'continuous','domain':(0.10,0.25)},
          {'name':"layer2_m",'type':'continuous','domain':(4,4.8)},
          {'name':"layer2_std","type":"continuous","domain":(0.4,0.6)},
          {'name':'layer3_m','type':"continuous",'domain':(4,4.8)},
          {'name':'layer3_std','type':'continuous','domain':(0.2,0.5)},
          {'name':'layer4_m','type':'continuous','domain':(5.8,6.5)},
          {'name':'layer4_std','type':'continuous','domain':(0.5,1)},
          {'name':'layer1_num','type':'discrete','domain':(4,5,6,7,8,9,10)},
          {'name':'layer2_num','type':'discrete','domain':(5,6,15,16,18,20,22,25,30)},
          {'name':'layer3_num','type':'discrete','domain':(1,3,5,7,9,10,12,14)},
          {'name':'layer4_num','type':'discrete','domain':(20,21)},
          {'name':'layer2_F','type':'continuous','domain':(1,1)},
          {'name':'layer3_F','type':'continuous','domain':(0,0)},
          {'name':'layer4_F','type':'continuous','domain':(1,1)},
          {'name':'layer2_U','type':'continuous','domain':(0,0)},
          {'name':'layer3_U','type':'continuous','domain':(1,1)},
          {'name':'layer4_U','type':'continuous','domain':(0,0)},
          {'name':'uniform_num','type':'discrete','domain':(3,4,5,6,7,8)}]#F2U3
space=GPyOpt.Design_space(space=bounds_conf)
data_init=config['dataset']['conf_num']
configuration=initial_design('random',space,data_init)

fig,ax=plt.subplots()
progress=tqdm(configuration,total=len(configuration))
save_con=dict()
save_gr=[]
save_gr1=[]
for con in progress:
    #print(con)
    progress.set_postfix_str(f"config.{con_num}")
    uniform_background(1)
    normal_distro_S1(con[0], con[1])
    normal_distro_S2(con[2], con[3])
    normal_distro_S3(con[4], con[5])
    normal_distro_S4(con[6], con[7])
    config_layer_uni_s=['F']
    config_layer1_s=config['structure']['layer1_symbol']#a list fixed
    config_layer2_s=config['structure']['layer2_symbol']#a list fixed
    config_layer3_s=config['structure']['layer3_symbol']# a list fixed
    config_layer4_s=config['structure']['layer4_symbol']# a list fixed

    print(config_layer1_s,config_layer2_s,config_layer3_s,config_layer4_s)

    n_frames = n_frames
    # n_atoms_outer=11
    # n_cl=con[1]
    # n_zn=2
    # n_frac=.5
    # n_cl_e=round(n_frac*n_atoms_outer)
    # n_k=n_atoms_outer-n_cl_e
    print(con[12],con[13],con[14])
    layer2_s_num=atom_number(config_layer2_s,con[9],[con[12],con[15]])
    layer3_s_num=atom_number(config_layer3_s,con[10],[con[13],con[16]])
    layer4_s_num=atom_number(config_layer4_s,con[11],[con[14],con[17]])
    print(f"layer1_s_num={con[8]}")
    print(f"layer2_s_num={layer2_s_num}")
    print(f"layer3_s_num={layer3_s_num}")
    print(f"layer4_s_num={layer4_s_num}")
    print(f"between layer:{con[18]}")
    #print(config_layer2_s,config_layer3_s,config_layer4_s)
    #print(con[8])
    #print(config_layer1_s)
    #for i in range(len(config_layer2_s)):
        #if layer2_s_num[i]==0:
           # del layer2_s_num[i]
           #layer2_s_num.pop(i)
           #config_layer2_s.pop(i)
    #print(layer2_s_num)
    #print(config_layer2_s)
    
    #for i in range(len(config_layer3_s)):
        #if layer3_s_num[i]==0:
           # del layer2_s_num[i]
           #layer3_s_num.pop(i)
           #config_layer3_s.pop(i)
    #print(layer3_s_num)
    #print(config_layer3_s)

    #for i in range(len(config_layer4_s)):
        #if layer4_s_num[i]==0:
           # del layer2_s_num[i]
           #layer4_s_num.pop(i)
           #config_layer4_s.pop(i)
    #print(layer4_s_num)
    #print(config_layer4_s)
    sum_atom=np.int32(np.sum([con[8]])+np.sum(layer2_s_num)+np.sum(layer3_s_num)+np.sum(layer4_s_num)+np.sum(con[18]))


    partial_bonds = []
    #sum_atom=0
    for i in range(n_frames):
        frame_list = []
        frame_list.append(S1.rvs(int(con[8])))
        frame_list.append(con[0]+(con[2]-con[0])*S0.rvs(int(con[18]))) #add a smooth buffer for the connection of two gaussian
        for j in range(len(layer2_s_num)):
            #sum_atom+=int(layer2_s_num[j])
            if layer2_s_num[j]!=0:
                frame_list.append(S2.rvs(int(layer2_s_num[j])))

        for j in range(len(layer3_s_num)):
            #sum_atom+=int(layer3_s_num[j])
            if layer3_s_num[j]!=0:
                frame_list.append(S3.rvs(int(layer3_s_num[j])))

        for j in range(len(layer4_s_num)):
            #sum_atom+=int(layer4_s_num[j])
            if layer4_s_num[j]!=0:
                frame_list.append(S4.rvs(int(layer4_s_num[j])))
        partial_bonds.append(frame_list)


    #print(f"\n\npartial_bonds ={(partial_bonds)}\n\n")
        #print(f"sum_atom={sum_atom}")
        #print(f'frame_list={frame_list}')
        #print(partial_bonds)
    #print(len(partial_bonds))
    #all_bonds = np.concatenate((np.array([l[0] for l in partial_bonds]).flatten(),np.array([l[1] for l in partial_bonds]).flatten(),np.array([l[2] for l in partial_bonds]).flatten(),np.array([l[3] for l in partial_bonds]).flatten()))

    #plt.hist(all_bonds, bins=100)
    #plt.show()
    rdf=[]
    #rdf1=[]
    for i in range(len(partial_bonds)):
        temp=[]
        for j in range(len(partial_bonds[i][0])):
            temp.append(partial_bonds[i][0][j])
        #rdf1.append(make_rdf_feff(temp))
        for j in range(len(partial_bonds[i][1])):
            temp.append(partial_bonds[i][1][j])
        for j in range(len(partial_bonds[i][2])):
            temp.append(partial_bonds[i][2][j])
        for j in range(len(partial_bonds[i][3])):
            temp.append(partial_bonds[i][3][j])
        for j in range(len(partial_bonds[i][4])):
            temp.append(partial_bonds[i][4][j])
        r,gr,rho=make_rdf_feff(temp)
        rdf.append(gr)

    #print(save_con)
    
    save_con[con_num]={"layer1_mean":con[0],
                      "layer1_std":con[1],
                      'layer1_layer2':con[18],
                      "layer2_mean":con[2],
                      "layer2_std":con[3],
                      "layer3_mean":con[4],
                      "layer3_std":con[5],
                      "layer4_mean":con[6],
                      "layer4_std":con[7],
                      "layer1_comp":config_layer1_s,
                      "layer1_comp_num":np.int32([con[8]]),
                      "layer2_comp":config_layer2_s,
                      "layer2_comp_num":np.int32(layer2_s_num),
                      "layer3_comp":config_layer3_s,
                      "layer3_comp_num":np.int32(layer3_s_num),
                      "layer4_comp":config_layer4_s,
                      "layer4_comp_num":np.int32(layer4_s_num),
                      'Density':np.round(rho,6)
                      }


    #print(f"{save_con[con_num]}")
    n_atoms_total=sum_atom
    #print(f"n_atoms={n_atoms_total}")



    optimized_results = []
    
    optimize_func=partial(optimization,partial_bonds)
    #print(partial_bonds)
    np.savetxt("rmesh.txt",r,delimiter=',')
    with open("output.dat",'a') as file1:
        file1.write(f"configuration.{con_num}:{save_con[con_num]}\n")
    try:
    #for frame in tqdm(range(len(partial_bonds)),total=len(partial_bonds)):
    #    optimized_results.append(optimize_func(frame))
    #k=0
        with confu.ProcessPoolExecutor(max_workers=config["mpi"]["tasks"]) as executor:
            jobs=[executor.submit(optimize_func,frame) for frame in range(len(partial_bonds))]
            for job in futures.as_completed(jobs):
                #print(k)
                #try:
                saved=job.result()
                optimized_results.append(saved)
                #k=k+1
                #except:
                #    continue
    #print(f"len_optimized_results:{len(optimized_results)}")
    #except ValueError:
    except Exception as e:
       print("bad structure!")
       with open("output.dat",'a') as file1:
            file1.write(f"{e}\n")
            save_con.pop(con_num)

    rdf=np.array(rdf).mean(axis=0)
    #rdf1=np.array(rdf1).mean(axis=0)
    
    #plot
    ax.plot(r,rdf,color=f"C{con_num}",label=f"config {con_num}")

    
    
    save_gr.append(rdf)
    #save_gr1.append(rdf1)





    all_results = np.array(optimized_results)
    atom_ids=[]
    atom_symbol_dic_str_to_int=config['structure']['atom_dict']
    for i in range(len(config_layer1_s)):
        for j in range(int(con[8])):
            atom_ids.append([config_layer1_s[i]])
    for i in range(len(config_layer_uni_s)):
        for j in range(int(con[18])):
            atom_ids.append([config_layer_uni_s[i]])
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
        os.makedirs('input')
    except:
        pass
    for i, stuff in enumerate(master_set_x):

        out=make_xyz([[atom_symbol_dic[str(int(s))] for s in stuff[:,0]],stuff[:,1:]])
        with open(f"input/config_{con_num}_frame_{i}_site_0.xyz", "w") as f:
            f.write(f"{len(stuff)}\n")
            f.write(f"UF4_test\n")
            f.write(out)
    #pd.DataFrame(save_con).T.to_csv("configuration.csv")
    #pd.DataFrame(save_gr).to_csv("gr.csv")

    con_num+=1
ax.set_xlabel('r($\AA$)')
ax.set_ylabel('g(r)')
ax.legend(frameon=False)
plt.savefig("gr.png")
pd.DataFrame(save_con).T.to_csv("configuration.csv")
pd.DataFrame(save_gr).to_csv("gr.csv")
#pd.DataFrame(save_gr1).to_csv("gr1.csv")
