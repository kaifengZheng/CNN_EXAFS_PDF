import os
from functools import partial


from .data_helper import *
from .data_loader import *

class NNGrDataModule(pl.LightningDataModule):
    def __init__(self,data_dir: str = './dataset', 
                 spectra_data_path: str = "spectra.csv",
                 gr_data_path=['gr.csv','gr_U.csv','gr_F.csv'],
                 kmesh_path='kmesh.txt',
                 rmesh_path='rmesh.txt',
                 kmesh_bound=[2,8],
                 rmesh_bound=[1,5],
                 training_noise_level=0.1,
                 kmesh_points=200,
                 rmesh_points=100,
                 k_win_range=[3,7],
                 random_seed:int=0,
                 batch=128,
                 kweight=2,
                 dx=2,
                 deltaE=0,
                 window_func='hanning',
                 concate=True,
                 concate_details={
                        'gr_F_range':[1.5,5.5],
                        'gr_U_range':[3.5,5.5],
                        # 'gr_F_num':200,
                        # 'gr_U_num':100                  
                 },
                 partial_gr=False):
        super().__init__()
        self.data_dir = data_dir
        self.spectra_data_path = spectra_data_path
        self.gr_data_path=gr_data_path
        self.kmesh_path=kmesh_path
        self.rmesh_path=rmesh_path
        self.kmesh_bound=kmesh_bound
        self.rmesh_bound=rmesh_bound
        self.random_seed=random_seed
        self.batch=batch
        self.training_noise_level=training_noise_level
        self.kmesh_points=kmesh_points
        self.rmesh_points=rmesh_points
        self.kweight=kweight
        self.dx=dx
        self.window_func=window_func
        self.k_win_range=k_win_range
        self.concate=concate
        self.partial_gr=partial_gr
        self.deltaE=deltaE
        if self.concate==True:
            self.gr_F_range=concate_details['gr_F_range']
            self.gr_U_range=concate_details['gr_U_range']
            self.partial_gr=True


    def prepare_data_train(self):  
        spectrum=pd.read_csv(os.path.join(self.data_dir,self.spectra_data_path))

        
        if self.concate==False:
            gr=pd.read_csv(os.path.join(self.data_dir,self.gr_data_path[0]),index_col=0) 
        else:
            gr=pd.read_csv(os.path.join(self.data_dir,self.gr_data_path[0]),index_col=0) 
            gr_U=pd.read_csv(os.path.join(self.data_dir,self.gr_data_path[1]),index_col=0)
            gr_F=pd.read_csv(os.path.join(self.data_dir,self.gr_data_path[2]),index_col=0)
        self.ori_kmesh=np.loadtxt(os.path.join(self.data_dir,self.kmesh_path))
        self.ori_rmesh=np.loadtxt(os.path.join(self.data_dir,self.rmesh_path))
        
        if self.concate==True:
            spectrum,[gr,gr_U,gr_F]=drop_bad_data(spectrum,[gr,gr_U,gr_F])
            if self.gr_F_range[0]<self.rmesh_bound[0]:
                self.gr_F_range[0]=self.rmesh_bound[0]
            if self.gr_F_range[1]>self.rmesh_bound[1]:
                self.gr_F_range[1]=self.rmesh_bound[1]
            if self.gr_U_range[0]<self.rmesh_bound[0]:
                self.gr_U_range[0]=self.rmesh_bound[0]
            if self.gr_U_range[1]>self.rmesh_bound[1]:
                self.gr_U_range[1]=self.rmesh_bound[1]

            gr_train,self.r_dict=data_concate(gr_U,gr_F,gr_F_start=self.gr_F_range[0],gr_F_end=self.gr_F_range[1],gr_U_start=self.gr_U_range[0],gr_U_end=self.gr_U_range[1],rmesh=self.ori_rmesh,rmesh_bound=self.rmesh_bound,rmesh_points=self.rmesh_points)
            self.rmesh_points=self.r_dict['total_r_points']
            self.rmesh_new=self.r_dict['rmesh_new']
            if self.window_func=='hanning':
               self.sample,self.kmesh_new,self.w=interp_sample_gr_concate(spectrum,gr_train,self.ori_kmesh,self.kmesh_bound,self.kmesh_points,self.kweight,
                                                                dx=self.dx,window_func=self.window_func,k_win_range=self.k_win_range)
            else:
               self.sample,self.kmesh_new=interp_sample_gr_concate(spectrum,gr_train,self.ori_kmesh,self.kmesh_bound,self.kmesh_points,self.kweight,
                                                                dx=self.dx,window_func=self.window_func,k_win_range=self.k_win_range)
        else:
             spectrum,gr=drop_bad_data(spectrum,gr)
             if self.window_func=='hanning':
               self.sample,self.kmesh_new,self.rmesh_new,self.w=interp_sample(spectrum,gr,self.ori_kmesh,self.ori_rmesh,self.kmesh_bound,self.rmesh_bound,self.kmesh_points,self.rmesh_points,
                                                                dx=self.dx,window_func=self.window_func,kweight=self.kweight,k_win_range=self.k_win_range)
             else:
               self.sample,self.kmesh_new,self.rmesh_new=interp_sample(spectrum,gr,self.ori_kmesh,self.ori_rmesh,self.kmesh_bound,self.rmesh_bound,self.kmesh_points,self.rmesh_points,
                                                                dx=self.dx,window_func=self.window_func,kweight=self.kweight,k_win_range=self.k_win_range)
        

    
    def prepare_test_data(self):
        test_loader=self.val_dataloader()
        test_x,test_y=next(iter(test_loader))
        return test_x,test_y
    
    def prepare_predict_data(self,spectrum_data:np.array,kmesh_test:np.array):
        spectrum=[]
        spectrum_data=spectrum_data.reshape(-1,len(kmesh_test))
        for i in range(spectrum_data.shape[0]):
            kdata=np.array((kmesh_test**self.kweight*spectrum_data[i]))
            if self.window_func=='hanning':
                kdata,w=kdata_cut_win(kmesh_test,kdata,self.k_win_range,dx=self.dx,window=self.window_func)
            spectrum.append(interp1d(np.round(kmesh_test,3),np.array(kdata))(np.round(self.kmesh_new,3)))
        spectrum=np.array(spectrum).reshape(-1,1,len(self.kmesh_new))
        spectrum=torch.from_numpy(spectrum.astype(np.float32))
        if self.window_func=='hanning':
            w=interp1d(np.round(kmesh_test,3),w)(np.round(self.kmesh_new,3))
            return spectrum,w
        else:
            return spectrum
    def prepare_predict_data_from_BF(self,spectrum_data:np.array,kmesh_test:np.array):
        spectrum=[]
        spectrum_data=spectrum_data.reshape(-1,len(kmesh_test))
        for i in range(spectrum_data.shape[0]):
            kdata=np.array((spectrum_data[i]))
            if self.window_func=='hanning':
                kdata,w=kdata_cut_win(kmesh_test,kdata,self.k_win_range,dx=self.dx,window=self.window_func)
            spectrum.append(interp1d(np.round(kmesh_test,3),np.array(kdata))(np.round(self.kmesh_new,3)))
        spectrum=np.array(spectrum).reshape(-1,1,len(self.kmesh_new))
        spectrum=torch.from_numpy(spectrum.astype(np.float32))
        if self.window_func=='hanning':
            w=interp1d(np.round(kmesh_test,3),w)(np.round(self.kmesh_new,3))
            return spectrum,w
        else:
            return spectrum
        
    def post_prediction(self,y_hat:np.array):
         y_hat=torch.nn.functional.relu(y_hat)
         y_hat=y_hat.detach().numpy()
         if self.concate==True:
             gr_F,gr_U,gr_total,rmesh_total=split_gr(self.r_dict,y_hat)
             return gr_F,gr_U,gr_total,rmesh_total
         else:   
                return y_hat,self.rmesh_new


        
    def setup(self,stage:str):
        if stage=='fit':
            self.train_data_func=partial(data_generator,self.sample,self.kmesh_new,self.training_noise_level)
            self.val_data_func=partial(val_generator,self.sample,self.kmesh_new)
        elif stage=='test':
            self.test_data_func=partial(val_generator,self.sample,self.kmesh_new)
            
    def train_dataloader(self):
        return GeneratorDataset(self.train_data_func,self.batch)

    def val_dataloader(self):
        return GeneratorDataset(self.val_data_func,self.batch)

    def test_dataloader(self):
        return GeneratorDataset(self.val_data_func,self.batch)
    def __str__(self):
        if self.concate==True:
            return f"data_dir={self.data_dir}\n \
                    spectra_data_path={self.spectra_data_path}\n \
                    gr_data_path={self.gr_data_path}\n \
                    kmesh_path={self.kmesh_path}\n \
                    rmesh_path={self.rmesh_path}\n \
                    kmesh_bound={self.kmesh_bound}\n \
                    rmesh_bound={self.rmesh_bound}\n \
                    random_seed={self.random_seed}\n \
                    batch={self.batch}\n \
                    training_noise_level={self.training_noise_level}\n \
                    kmesh_points={self.kmesh_points}\n \
                    rmesh_points={self.rmesh_points}\n \
                    kweight={self.kweight}\n \
                    dx={self.dx}\n \
                    window_func={self.window_func}\n \
                    k_win_range={self.k_win_range}\n \
                    concate={self.concate}\n \
                    gr_F_range={self.gr_F_range}\n \
                    gr_U_range={self.gr_U_range}\n  \
                    partial_gr={self.partial_gr}\n"
        else:
            return f"data_dir={self.data_dir}\n \
                    spectra_data_path={self.spectra_data_path}\n \
                    gr_data_path={self.gr_data_path}\n \
                    kmesh_path={self.kmesh_path}\n \
                    rmesh_path={self.rmesh_path}\n \
                    kmesh_bound={self.kmesh_bound}\n \
                    rmesh_bound={self.rmesh_bound}\n \
                    random_seed={self.random_seed}\n \
                    batch={self.batch}\n \
                    training_noise_level={self.training_noise_level}\n \
                    kmesh_points={self.kmesh_points}\n \
                    rmesh_points={self.rmesh_points}\n \
                    kweight={self.kweight}\n \
                    dx={self.dx}\n \
                    window_func={self.window_func}\n \
                    k_win_range={self.k_win_range}\n \
                    concate={self.concate}\n \
                    partial_gr={self.partial_gr}\n"

