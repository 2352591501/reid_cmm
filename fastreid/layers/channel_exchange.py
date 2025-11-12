import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class CE_module(nn.Module):
    def __init__(self, probability=0.5, threshold=0.3):
        super(CE_module, self).__init__()
        self.probability = probability
        self.threshold = threshold

    def forward(self, CA, feature_map):
        if random.uniform(0, 1) >= self.probability:
            bs = feature_map.size(0) // 2
            feature_map_ir = feature_map[:bs]
            CA_ir = CA[:bs]
            CA_rgb = CA[bs:]
            feature_map_rgb = feature_map[bs:]
            CE_E_ir = (CA_ir < self.threshold)
            CE_E_ir = CE_E_ir.view(CE_E_ir.size(0), CE_E_ir.size(1))
            CE_E_rgb = (CA_rgb < self.threshold)
            CE_E_rgb = CE_E_ir.view(CE_E_rgb.size(0), CE_E_rgb.size(1))
            CE_N_ir = (CE_E_ir == False)
            CE_N_rgb = (CE_E_rgb == False)
            x1, x2 = torch.zeros_like(feature_map_ir), torch.zeros_like(feature_map_rgb)
            x1[CE_N_ir] = feature_map_ir[CE_N_ir]
            x1[CE_E_ir] = feature_map_rgb[CE_E_ir]
            x2[CE_N_rgb] = feature_map_rgb[CE_N_rgb]
            x2[CE_E_rgb] = feature_map_ir[CE_E_rgb]
            return torch.cat((x1, x2), dim=0)
        else:
            return feature_map