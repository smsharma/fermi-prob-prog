import numpy as np
import torch
import torch.nn as nn



def distance_corr(A, B):
    '''
    Calculate the Distance Correlation between the two vectors. https://en.wikipedia.org/wiki/Distance_correlation
    Value of 0 implies independence. A and B can be vectors of different length.
    :param A:    vector A of shape (num_samples, sizeA)
    :param B:    vector B of shape (num_samples, sizeB)
    :return:     the distance correlation between A and B
    '''
    a = _distance_matrix(A)
    b = _distance_matrix(B)
    dist_cov2_ab = torch.clamp(torch.div(torch.sum(a * b), A.shape[0] * A.shape[0]), 1e-10, 1e10)
    dist_cov2_aa = torch.clamp(torch.div(torch.sum(a * a), A.shape[0] * A.shape[0]), 1e-10, 1e10)
    dist_cov2_bb = torch.clamp(torch.div(torch.sum(b * b), A.shape[0] * A.shape[0]), 1e-10, 1e10)
    dist_var_prod= torch.clamp(torch.sqrt(dist_cov2_aa) * torch.sqrt(dist_cov2_bb), 1e-10, 1e10)
    dist_cor = torch.div(torch.sqrt(dist_cov2_ab), torch.sqrt(dist_var_prod))
    return dist_cor

def _distance_matrix(x):
    '''
    Input: x is a Nxd matrix
        y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x**2).sum(1).view(-1, 1)
    y_t = torch.transpose(x, 0, 1)
    y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)

    dist = torch.clamp(dist, 0.0, np.inf)
    dist = torch.sqrt(torch.clamp(dist, 1e-10, 1e10))
    rows_mean = torch.mean(dist, 0, True)
    columns_mean = torch.mean(dist, 1, True)
    distance = dist - rows_mean - columns_mean + torch.mean(dist)
    return distance
    
# def distance_corr(var_1,var_2,normedweight,power=1):
#     """var_1: First variable to decorrelate (eg mass)
#     var_2: Second variable to decorrelate (eg classifier output)
#     normedweight: Per-example weight. Sum of weights should add up to N (where N is the number of examples)
#     power: Exponent used in calculating the distance correlation
    
#     va1_1, var_2 and normedweight should all be 1D torch tensors with the same number of entries
    
#     Usage: Add to your loss function. total_loss = BCE_loss + lambda * distance_corr
#     """
    
    
#     xx = var_1.view(-1, 1).repeat(1, len(var_1)).view(len(var_1),len(var_1))
#     yy = var_1.repeat(len(var_1),1).view(len(var_1),len(var_1))
#     amat = (xx-yy).abs()

#     xx = var_2.view(-1, 1).repeat(1, len(var_2)).view(len(var_2),len(var_2))
#     yy = var_2.repeat(len(var_2),1).view(len(var_2),len(var_2))
#     bmat = (xx-yy).abs()

#     amatavg = torch.mean(amat*normedweight,dim=1)
#     Amat=amat-amatavg.repeat(len(var_1),1).view(len(var_1),len(var_1))\
#         -amatavg.view(-1, 1).repeat(1, len(var_1)).view(len(var_1),len(var_1))\
#         +torch.mean(amatavg*normedweight)

#     bmatavg = torch.mean(bmat*normedweight,dim=1)
#     Bmat=bmat-bmatavg.repeat(len(var_2),1).view(len(var_2),len(var_2))\
#         -bmatavg.view(-1, 1).repeat(1, len(var_2)).view(len(var_2),len(var_2))\
#         +torch.mean(bmatavg*normedweight)

#     ABavg = torch.mean(Amat*Bmat*normedweight,dim=1)
#     AAavg = torch.mean(Amat*Amat*normedweight,dim=1)
#     BBavg = torch.mean(Bmat*Bmat*normedweight,dim=1)

#     if(power==1):
#         dCorr=(torch.mean(ABavg*normedweight))/torch.sqrt((torch.mean(AAavg*normedweight)*torch.mean(BBavg*normedweight)))
#     elif(power==2):
#         dCorr=(torch.mean(ABavg*normedweight))**2/(torch.mean(AAavg*normedweight)*torch.mean(BBavg*normedweight))
#     else:
#         dCorr=((torch.mean(ABavg*normedweight))/torch.sqrt((torch.mean(AAavg*normedweight)*torch.mean(BBavg*normedweight))))**power
    
#     return dCorr

