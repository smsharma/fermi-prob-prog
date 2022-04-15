import torch
import torch.nn as nn

class DiffeomorphicCounterfactual(nn.Module):
    def __init__(self, z0, device='cuda'):
        """
        In the constructor we instantiate four parameters and assign them as
        member parameters.
        """
        super().__init__()
        self.z = nn.ParameterList(z0).to(device)

        self.device = device

    def forward(self, model_gen, model_inf, mask, param_idx=0):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        reco = model_gen.flow.reverse(self.z, reconstruct=True, quant_int=False) + 1e-6
        return torch.softmax(model_inf.resnet((reco.log10())[0] * torch.Tensor(~mask).to(self.device))[:, :8], dim=-1)[0, param_idx]


class GenerateCounterfactuals:
    def __init__(self, x0, model_gen, model_inf, mask, device='cuda', sigma=1.):

        self.model_gen = model_gen
        self.model_inf = model_inf
        
        _, _, z0 = model_gen.flow(x0.to(device))
        out = model_inf.resnet(((x0[0] + 1e-6).log10() * torch.Tensor(~mask)).to(device))

        self.mu, self.logvar = torch.chunk(out, 2, -1)
        self.mu = torch.softmax(self.mu, dim=-1)
        
        print(self.mu)
        print(self.logvar.exp().sqrt())
        
        self.sigma = sigma
        self.device = device

        self.mask = mask

        self.z0 = z0

    def generate_counterfactual(self, i_param=0, cf_type=1, lr=0.1, print_every=10):
        
        z0_param = [nn.Parameter(zz.to(self.device)) for zz in self.z0.copy()]
        dc = DiffeomorphicCounterfactual(z0_param, device=self.device).to(self.device)
        
        optimizer = torch.optim.SGD(dc.parameters(), lr=lr)

        i = 0
        
        condition = True
        
        while condition:

            if cf_type == -1:
                y_pred = dc(self.model_gen, self.model_inf, self.mask, i_param)
                if i % print_every == 0:
                    print(y_pred)
            elif cf_type == 1:
                y_pred = -dc(self.model_gen, self.model_inf, self.mask, i_param)
                if i % print_every == 0:
                    print(-y_pred)
            else:
                raise NotImplementedError

            optimizer.zero_grad()
            y_pred.backward()
            optimizer.step()

            if cf_type == -1:
                condition = y_pred > (self.mu - self.sigma * self.logvar.exp().sqrt())[0, i_param]
            elif cf_type == 1:
                condition = y_pred > -(self.mu + self.sigma * self.logvar.exp().sqrt())[0, i_param]
            else:
                raise NotImplementedError

            i += 1
            
        z_cf = dc.z
        x_cf = self.model_gen.flow.reverse(dc.z, reconstruct=True, quant_int=True, quant_type='round')[0,0].cpu().detach().numpy()
            
        return x_cf, z_cf