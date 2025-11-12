import torch
from fastreid.modeling.backbones.resnet_for_parsing import resnet101, FineTuneBlock
from collections import OrderedDict

# prasing network
parsing_net = resnet101(num_classes=7, pretrained=None)
state_dict = torch.load(
    '/home/amax/zn/Self-Correction-Human-Parsing-master/pretrain_model/exp-schp-201908270938-pascal-person-part.pth')[
    'state_dict']
new_state_dict = OrderedDict()
fine_tune_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:]  # remove `module.`
    new_state_dict[name] = v
    if 'context' in name:
        fine_tune_state_dict[name] = v
    if 'decoder' in name:
        fine_tune_state_dict[name] = v
    if 'edge' in name:
        fine_tune_state_dict[name] = v
    if 'fushion' in name:
        fine_tune_state_dict[name] = v
parsing_net.load_state_dict(new_state_dict)
parsing_net.cuda()
parsing_net.eval()
fine_tune_net = FineTuneBlock(num_classes=7)
fine_tune_net.load_state_dict(fine_tune_state_dict)
fine_tune_net.cuda()
fine_tune_net.eval()



breakpoint()