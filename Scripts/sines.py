import torch
import numpy as np
from torch.utils.data import Dataset
from Scripts.utility_func import set_seed
from tqdm.auto import tqdm

class CreateDataset(Dataset):
    def __init__(self, data1, data2, period='train'):
        super().__init__()
        self.season = data1
        self.trend = data2
        self.samples = data1 + data2
        self.period = period

    def __getitem__(self, ind):
        x = self.samples[ind, :, :]
        if self.period == 'train':
            return torch.from_numpy(x).float()
        
        t = self.trend[ind, :, :]
        s = self.season[ind, :, :]
        return torch.from_numpy(x).float(), torch.from_numpy(t).float(),\
               torch.from_numpy(s).float()

    def __len__(self):
        return self.samples.shape[0]
    
    def data_generation(no, seq_len, dim, seed=123):
        st0 = np.random.get_state()
        set_seed(seed)
        data_1, data_2 = list(), list()
        for i in tqdm(range(0, no), total=no, desc="Sampling Data"):
            local, glob = list(), list()
            for k in range(dim):
                coeff_a = np.random.uniform(0, 1)            
                coeff_b = np.random.uniform(0, 1)
                coeff_c = np.random.uniform(0.1, 0.5)
                coeff_d = np.random.uniform(1, 3)

                temp_data1 = [coeff_c * (np.sin(0.2 * np.pi * j * (k+1) + coeff_a) + 2 * np.sin(0.1 * np.pi * j * (k+1) + coeff_b)) for j in range(seq_len)]

                signal = np.random.uniform(0, 1)
                if signal > 0.5:
                    temp_data2 = [coeff_d * np.sin(0.001 * np.pi * j) for j in range(seq_len)]
                else:
                    temp_data2 = [- coeff_d * np.sin(0.001 * np.pi * j) for j in range(seq_len)]

                local.append(temp_data1)
                glob.append(temp_data2)

            local = np.transpose(np.asarray(local))
            data_1.append(local)
            glob = np.transpose(np.asarray(glob))
            data_2.append(glob)

        np.random.set_state(st0)
        data_1 = np.array(data_1)
        data_2 = np.array(data_2)
        return data_1, data_2