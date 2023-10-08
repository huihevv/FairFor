import torch
import random
import numpy as np

def init_seed(seed):
    '''
    Disable cudnn to maximize reproducibility
    '''
    torch.cuda.cudnn_enabled = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def print_model_parameters(model1, model2, only_num = True):
    print('*****************Model Parameter*****************')
    if not only_num:
        for name, param in model1.named_parameters():
            print(name, param.shape, param.requires_grad)
    if not only_num:
        for name, param in model2.named_parameters():
            print(name, param.shape, param.requires_grad)
    total_num_1 = sum([param.nelement() for param in model1.parameters()])
    total_num_2 = sum([param.nelement() for param in model2.parameters()])
    print('Total params num: %.2fM' % ((total_num_1 + total_num_2) / 1e6))
    # print('Total params num: %.2fM' % (total_num_1 + total_num_2))
    print('*****************Finish Parameter****************')

def get_memory_usage(device):
    allocated_memory = torch.cuda.memory_allocated(device) / (1024*1024.)
    cached_memory = torch.cuda.memory_cached(device) / (1024*1024.)
    return allocated_memory, cached_memory