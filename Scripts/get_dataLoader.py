import torch
from Scripts.utility_func import create_instance_from_config

def get_dataloader_common(config, dataset_key, batch_size_key, shuffle, save_dir=None):
    batch_size = config['dataloader'][batch_size_key]
    config['dataloader'][dataset_key]['params']['output_dir'] = save_dir
    dataset = create_instance_from_config(config['dataloader'][dataset_key])

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=True, sampler=None, drop_last=shuffle)

    dataload_info = {
        'dataloader': dataloader,
        'dataset': dataset
    }

    return dataload_info

#dataloader for unconditional sampling
def get_dataloader(config, save_dir=None):
    return get_dataloader_common(config, 'train_dataset', 'batch_size', True, save_dir)

#data loader for conditional sampling
def get_dataloader_cond(config, mode=None, pred_len=None, save_dir=None):
    if mode == 'predict':
        config['dataloader']['test_dataset']['params']['predict_length'] = pred_len

    return get_dataloader_common(config, 'test_dataset', 'sample_size', False, save_dir)


if __name__ == '__main__':
    pass