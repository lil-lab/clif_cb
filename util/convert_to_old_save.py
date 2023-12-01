from collections import OrderedDict
import torch
import sys

load_from = sys.argv[1]
save_to = sys.argv[2]

model_save = torch.load(load_from, map_location=torch.device('cpu'))


def key_changer(key):
    if '_lingunet' in key and '_convolution_layers' in key and 'weight' in key and '_weight' not in key or key == '_lingunet._final_deconv.weight':
        new_name = key.replace('weight', '_weight')
        return new_name
    return key


new_orderedict = OrderedDict()
for param_name, param in model_save.items():
    new_name = key_changer(param_name)
    new_orderedict[new_name] = param
        

torch.save(new_orderedict, save_to, _use_new_zipfile_serialization=False)

print(save_to)
