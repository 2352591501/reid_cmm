import torch
import torch.nn as nn

class Interpolator(nn.Module):
    def __init__(self):
        super(Interpolator, self).__init__()
        self.beta = torch.distributions.Beta(0.1, 0.1)

    def forward(self, CA, x):
        '''
        bs_split = feature_map.size(0) // 2
        feature_map_ir = feature_map[:bs_split]
        CA_ir = CA[:bs_split]
        feature_map_rgb = feature_map[bs_split:]
        CA_rgb = CA[bs_split:]
        alpha = torch.sigmoid(-99999 * (CA_ir + CA_rgb - 0.6))
        feature_map_ir = feature_map_ir + alpha * (feature_map_rgb - feature_map_ir.detach())
        feature_map_rgb = feature_map_rgb + alpha * (feature_map_ir - feature_map_rgb.detach())

        return torch.cat((feature_map_ir, feature_map_rgb), dim=0)
        '''
        B, C, W, H = x.size(0), x.size(1), x.size(2), x.size(3)
        x_view = x.view(B, C, -1)
        ## sort input vectors.
        value_x, index_x = torch.sort(x_view)

        # split into two halves and swap the order
        perm = torch.arange(B - 1, -1, -1)  # inverse index
        perm_b, perm_a = perm.chunk(2)
        perm_b = perm_b[torch.randperm(B // 2)]
        perm_a = perm_a[torch.randperm(B // 2)]
        perm = torch.cat([perm_b, perm_a], 0)

        CA_perm = CA[perm]
        alpha = torch.sigmoid(-999999 * (CA + CA_perm - 0.6)).view(B, C, 1)

        inverse_index = index_x.argsort(-1)
        x_view_copy = value_x[perm].gather(-1, inverse_index) * alpha
        new_x = x_view + (x_view_copy - x_view.detach() * alpha)
        return new_x.view(B, C, W, H)