import numpy as np
import torch
import torch.nn.functional as F


class LpLoss(object):
    '''
    loss function with rel/abs Lp loss
    '''
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        #Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h**(self.d/self.p))*torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples, -1) - y.reshape(num_examples, -1) + 1e-5, self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples, -1) + 1e-5, self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms / (y_norms + 1e-5))
            else:
                return torch.sum(diff_norms / (y_norms + 1e-5))

        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.rel(x, y)

class LpLoss2(object):
    '''
    loss function with rel/abs Lp loss
    '''
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        #Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples, -1) - y.reshape(num_examples, -1) + 1e-5, self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples, -1) + 1e-5, self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms / (y_norms + 1e-5))
            else:
                return torch.sum(diff_norms / (y_norms + 1e-5))

        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.abs(x, y)


def SWE_CON(outputH, outputPXB, outputPYB, outputPX, outputPY, z, Rain, Manning, dt):
    # x, y, t = input_data[:,:,:,:,:1], input_data[:,:,:,:,1:2], input_data[:,:,:,:,2:3]
    h, qx, qy = outputH, outputPX, outputPY
    qxb, qyb = outputPXB, outputPYB
    z = torch.unsqueeze(z, dim=0)
    z = torch.unsqueeze(z, dim=0)
    s = h + z
    g = 9.8
    batchsize = h.size(0)
    # dx = (2*30.0*16-lb)/(ub-lb)
    # dy = (2*30.0*16-lb)/(ub-lb)
    dx = 30.0*16
    dy = 30.0*16
    # dt = (2*dt-lb)/(ub-lb)
    # dt = 1.0
    # Rain R
    R = Rain/(1000*60*60) # 1,l,m,n
    # Manning
    n = Manning # m,n
    # print('Manning', Manning.shape)
    # n = 0.05
    #H
    dsdxi_internal = (-s[:, :, 4:, :] + 8 * s[:, :, 3:-1, :] - 8 * s[:, :, 1:-3, :] + s[:, :, 0:-4, :]) / 12 / dx
    dsdxi_left = (-11 * s[:, :, 0:-3, :] + 18 * s[:, :, 1:-2, :] - 9 * s[:, :, 2:-1, :] + 2 * s[:, :, 3:, :]) / 6 / dx
    dsdxi_right = (11 * s[:, :, 3:, :] - 18 * s[:, :, 2:-1, :] + 9 * s[:, :, 1:-2, :] - 2 * s[:, :, 0:-3, :]) / 6 / dx
    dsdx = torch.cat((dsdxi_left[:, :, 0:2, :], dsdxi_internal, dsdxi_right[:, :, -2:, :]), 2)

    dsdyi_internal = (-s[:, :, :, 4:] + 8 * s[:, :, :, 3:-1] - 8 * s[:, :, :, 1:-3] + s[:, :, :, 0:-4]) / 12 / dy
    dsdyi_left = (-11 * s[:, :, :, 0:-3] + 18 * s[:, :, :, 1:-2] - 9 * s[:, :, :, 2:-1] + 2 * s[:, :, :, 3:]) / 6 / dy
    dsdyi_right = (11 * s[:, :, :, 3:] - 18 * s[:, :, :, 2:-1] + 9 * s[:, :, :, 1:-2] - 2 * s[:, :, :, 0:-3]) / 6 / dy
    dsdy = torch.cat((dsdyi_left[:, :, :, 0:2], dsdyi_internal, dsdyi_right[:, :, :, -2:]), 3)

    dhdt_internal = (-h[:, 4:, :, :] + 8 * h[:, 3:-1, :, :] - 8 * h[:, 1:-3, :, :] + h[:, 0:-4, :, :]) / 12 / dt
    dhdt_left = (-11 * h[:, 0:-3, :, :] + 18 * h[:, 1:-2, :, :] - 9 * h[:, 2:-1, :, :] + 2 * h[:, 3:, :, :]) / 6 / dt
    dhdt_right = (11 * h[:, 3:, :, :] - 18 * h[:, 2:-1, :, :] + 9 * h[:, 1:-2, :, :] - 2 * h[:, 0:-3, :, :]) / 6 / dt
    dhdt = torch.cat((dhdt_left[:, 0:2, :, :], dhdt_internal, dhdt_right[:, -2:, :, :]), 1)
    #qx
    dqxdxi_internal = (-qxb[:, :, 4:, :] + 8 * qxb[:, :, 3:-1, :] - 8 * qxb[:, :, 1:-3, :] + qxb[:, :, 0:-4, :]) / 12 / dx
    dqxdxi_left = (-11 * qxb[:, :, 0:-3, :] + 18 * qxb[:, :, 1:-2, :] - 9 * qxb[:, :, 2:-1, :] + 2 * qxb[:, :, 3:, :]) / 6 / dx
    dqxdxi_right = (11 * qxb[:, :, 3:, :] - 18 * qxb[:, :, 2:-1, :] + 9 * qxb[:, :, 1:-2, :] - 2 * qxb[:, :, 0:-3, :]) / 6 / dx
    dqxdx = torch.cat((dqxdxi_left[:, :, 0:2, :], dqxdxi_internal, dqxdxi_right[:, :, -2:, :]), 2)

    dqxdyi_internal = (-qxb[:, :,  :, 4:] + 8 * qxb[:, :, :, 3:-1] - 8 * qxb[:, :, :, 1:-3] + qxb[:, :, :, 0:-4]) / 12 / dy
    dqxdyi_left = (-11 * qxb[:, :, :, 0:-3] + 18 * qxb[:, :, :, 1:-2] - 9 * qxb[:, :, :, 2:-1] + 2 * qxb[:, :, :, 3:]) / 6 / dy
    dqxdyi_right = (11 * qxb[:, :, :, 3:] - 18 * qxb[:, :, :, 2:-1] + 9 * qxb[:, :, :, 1:-2] - 2 * qxb[:, :, :, 0:-3]) / 6 / dy
    dqxdy = torch.cat((dqxdyi_left[:, :, :, 0:2], dqxdyi_internal, dqxdyi_right[:, :, :, -2:,]), 3)

    dqxdt_internal = (-qx[:, 4:, :, :] + 8 * qx[:, 3:-1, :, :] - 8 * qx[:, 1:-3, :, :] + qx[:, 0:-4, :, :]) / 12 / dt
    dqxdt_left = (-11 * qx[:, 0:-3, :, :] + 18 * qx[:, 1:-2, :, :] - 9 * qx[:, 2:-1, :, :] + 2 * qx[:, 3:, :, :]) / 6 / dt
    dqxdt_right = (11 * qx[:, 3:, :, :] - 18 * qx[:, 2:-1, :, :] + 9 * qx[:, 1:-2, :, :] - 2 * qx[:, 0:-3, :, :]) / 6 / dt
    dqxdt = torch.cat((dqxdt_left[:, 0:2, :, :], dqxdt_internal, dqxdt_right[:, -2:, :, :]), 1)

    #qy
    dqydxi_internal = (-qyb[:, :, 4:, :] + 8 * qyb[:, :, 3:-1, :] - 8 * qyb[:, :, 1:-3, :] + qyb[:, :, 0:-4, :]) / 12 / dx
    dqydxi_left = (-11 * qyb[:, :, 0:-3, :] + 18 * qyb[:, :, 1:-2, :] - 9 * qyb[:, :, 2:-1, :] + 2 * qyb[:, :, 3:, :]) / 6 / dx
    dqydxi_right = (11 * qyb[:, :, 3:, :] - 18 * qyb[:, :, 2:-1, :] + 9 * qyb[:, :, 1:-2, :] - 2 * qyb[:, :, 0:-3, :]) / 6 / dx
    dqydx = torch.cat((dqydxi_left[:, :, 0:2, :], dqydxi_internal, dqydxi_right[:, :, -2:, :]), 2)

    dqydyi_internal = (-qyb[:, :, :, 4:] + 8 * qyb[:, :, :, 3:-1] - 8 * qyb[:, :, :, 1:-3] + qyb[:, :, :, 0:-4]) / 12 / dy
    dqydyi_left = (-11 * qyb[:, :, :, 0:-3] + 18 * qyb[:, :, :, 1:-2] - 9 * qyb[:, :, :, 2:-1] + 2 * qyb[:, :, :, 3:]) / 6 / dy
    dqydyi_right = (11 * qyb[:, :, :, 3:] - 18 * qyb[:, :, :, 2:-1] + 9 * qyb[:, :, :, 1:-2] - 2 * qyb[:, :, :, 0:-3]) / 6 / dy
    dqydy = torch.cat((dqydyi_left[:, :, :, 0:2], dqydyi_internal, dqydyi_right[:, :, :, -2:, ]), 3)

    dqydt_internal = (-qy[:, 4:, :, :] + 8 * qy[:, 3:-1, :, :] - 8 * qy[:, 1:-3, :, :] + qy[:, 0:-4, :, :]) / 12 / dt
    dqydt_left = (-11 * qy[:, 0:-3, :, :] + 18 * qy[:, 1:-2, :, :] - 9 * qy[:, 2:-1, :, :] + 2 * qy[:, 3:, :, :]) / 6 / dt
    dqydt_right = (11 * qy[:, 3:, :, :] - 18 * qy[:, 2:-1, :, :] + 9 * qy[:, 1:-2, :, :] - 2 * qy[:, 0:-3, :, :]) / 6 / dt
    dqydt = torch.cat((dqydt_left[:, 0:2, :, :], dqydt_internal, dqydt_right[:, -2:, :, :]), 1)

    #dvcosthtdy
    _EPSILON = 1e-6
    # hh = h.clone()
    # qxx = qx.clone()
    # qyy = qy.clone()
    # h.clamp(1e-6)
    # qx.clamp(1e-6)
    # qy.clamp(1e-6)
    # friction_x = g * (n ** 2) * ((qxx ** 2) ** 0.5) * qxx / (hh ** (7 / 3))
    # friction_y = g * (n ** 2) * ((qyy ** 2) ** 0.5) * qyy / (hh ** (7 / 3))
    friction_x = g * (n ** 2) * ((qx ** 2 + qy ** 2 + _EPSILON) ** 0.5) * qx / (h ** (7 / 3) + _EPSILON)
    friction_y = g * (n ** 2) * ((qx ** 2 + qy ** 2 + _EPSILON) ** 0.5) * qy / (h ** (7 / 3) + _EPSILON)
    # print('man', torch.mean(n))
    # print('friction_x', torch.mean(friction_x))
    # print('friction_y', torch.mean(friction_y))
    # print('dsdx', torch.mean(dsdx))
    # print('dqxdt', torch.mean(dqxdt))
    # print('dqydt', torch.mean(dqydt))
    # print('dsdy', torch.mean(dsdy))
    # print('dhdt', torch.mean(dhdt))
    # print('dqxdx', torch.mean(dqxdx))
    # print('dqydy', torch.mean(dqydy))
    # print('h', torch.min(h))
    # print('qy', torch.min(qy))
    # print('qx', torch.min(qx))


    eqnm = dhdt + dqxdx + dqydy - R
    eqnx = dqxdt + g*h*dsdx + friction_x
    eqny = dqydt + g*h*dsdy + friction_y
    # print('eqnm', torch.max(eqnm))
    # print('eqnx', torch.max(eqnx))
    # print('eqny', torch.max(eqny))

    return eqnm, eqnx, eqny
    # return eqnm, eqnx, eqny

def GeoPC_loss(input_data, outputH, outputU, outputV, data_condition, init_condition):

        #data_loss
    t0 = outputH.size(1)
    h_gt, u_gt, v_gt = data_condition[0], data_condition[1], data_condition[2]

    # loss_d = loss_h + loss_qx + loss_qy
    loss = LpLoss(size_average=True)
    h_c = outputH
    # h_g = torch.unsqueeze(h_gt, dim=0)
    loss_h = loss(h_c, h_gt)
    u_c = outputU
    # qx_g = torch.unsqueeze(qx_gt, dim=0)
    loss_u = loss(u_c, u_gt)
    v_c = outputV
    # qy_g = torch.unsqueeze(qy_gt, dim=0)
    loss_v = loss(v_c, v_gt)
    loss_d = loss_h + loss_u + loss_v

    # if i == 0:
    #     _EPSILON = 1e-6
    h_init = init_condition[0]
    h_cc = outputH[:, 0, :, :]
    #     h_c = torch.squeeze(h_c)
    loss_c = loss(h_cc, h_init)

    return loss_d, loss_c