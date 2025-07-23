import torch
import os
import numpy as np

################################################################
# Dataset class
################################################################
class flood_data(torch.utils.data.Dataset):
    def __init__(self, path_root, T_in, T_out=None, train=True, strategy="markov", std=0.0):
        self.markov = strategy == "markov"
        self.teacher_forcing = strategy == "teacher_forcing"
        self.one_shot = strategy == "oneshot"
        self.path_root = path_root
        # self.data = data[..., :(T_in + T_out)] if self.one_shot else data[..., :(T_in + T_out), :]
        self.data = []
        pt_files = [name for name in os.listdir(self.path_root) if name.endswith('.pt')]
        n = len(pt_files)
        print('Number of .pt files:', n)
        #for name in os.listdir(self.path_root):
        #    path_idx = os.path.join(self.path_root, name)
        self.data = [os.path.join(self.path_root, name) for name in pt_files]
        self.nt = T_in + T_out
        self.T_in = T_in
        self.T_out = T_out
        self.num_hist = 1 if self.markov else self.T_in
        self.train = train
        self.noise_std = std

    def log_transform(self, data, eps=1e-2):
        return torch.log(1 + data/eps)

    def __len__(self):
        if self.train:
            if self.markov:
                return len(self.data) * (self.nt - 1)
            if self.teacher_forcing:
                return len(self.data) * (self.nt - self.T_in)
        return len(self.data)

    def __getitem__(self, idx):
        if not self.train or not (self.markov or self.teacher_forcing): # full target: return all future steps
            pde_path = self.data[idx]
            # path_idx = os.path.join(path_root, str(idx) + ".pt")
            pde = torch.load(pde_path)
            # pde = pde.permute(1, 2, 0, 3)
            if self.one_shot:
                x = pde[..., :self.T_in, :3]
                mask = (x[..., 0:1] == 0.0)
                x[..., 1:2][mask] = 0.0
                x[..., 2:3][mask] = 0.0
                x[..., :3] = torch.nan_to_num(x[..., :3], nan=0.0)
                x1 = x.unsqueeze(-3).repeat([1, 1, self.T_out, 1, 1])
                # x_n = x.numpy()
                # x_n = np.nan_to_num(x_n, nan=-99999)
                # x_n = np.ma.masked_array(x_n, mask=(x_n < -2000))
                # x = torch.from_numpy(x_n).float()
                # nan_mask = torch.isnan(x)
                # contains_nan = torch.any(nan_mask)
                # print('contains_nan', contains_nan)
                p = pde[..., self.T_in:(self.T_in + self.T_out), 3:4]
                x2 = self.log_transform(p) / 10.0
                x2 = torch.unsqueeze(x2, dim = -1)
                z = pde[..., self.T_in:(self.T_in + self.T_out), 4:5]
                max_z = z.max()
                z_value = max_z + 30.0
                z = torch.nan_to_num(z, nan=z_value)
                x3 = torch.nn.functional.normalize(z)
                x3 = torch.unsqueeze(x3, dim=-1)
                x = torch.cat((x1, x2), dim=-1)
                x = torch.cat((x, x3), dim=-1)
                # x = x.unsqueeze(-3).repeat([1, 1, self.T_out, 1, 1])
                y = pde[..., self.T_in:(self.T_in + self.T_out), :3]
                mask_y = (y[..., 0:1] == 0.0)
                y[..., 1:2][mask_y] = 0.0
                y[..., 2:3][mask_y] = 0.0
                mask_tensor = ~torch.isnan(y)
                y = torch.nan_to_num(y, nan=0.0)
            else:
                x = pde[..., (self.T_in - self.num_hist):self.T_in, :3]
                mask = (x[..., 0:1] == 0.0)
                x[..., 1:2][mask] = 0.0
                x[..., 2:3][mask] = 0.0
                x[..., :3] = torch.nan_to_num(x[..., :3], nan=0.0)
                x1 = x.unsqueeze(-3).repeat([1, 1, self.T_out, 1, 1])
                # x_n = x.numpy()
                # x_n = np.nan_to_num(x_n, nan=-99999)
                # x_n = np.ma.masked_array(x_n, mask=(x_n < -2000))
                # x = torch.from_numpy(x_n).float()
                # nan_mask = torch.isnan(x)
                # contains_nan = torch.any(nan_mask)
                # print('contains_nan', contains_nan)
                p = pde[..., self.T_in:(self.T_in + self.T_out), 3:4]
                x2 = self.log_transform(p) / 10.0
                z = pde[..., self.T_in:(self.T_in + self.T_out), 4:5]
                max_z = z.max()
                z_value = max_z + 30.0
                z = torch.nan_to_num(z, nan=z_value)
                x3 = torch.nn.functional.normalize(z)
                x = torch.cat((x1, x2), dim=-1)
                x = torch.cat((x, x3), dim=-1)
                # x[..., :4] = self.log_transform(x[..., :4])
                y = pde[..., self.T_in:(self.T_in + self.T_out), :3]
                mask_y = (y[..., 0:1] == 0.0)
                y[..., 1:2][mask_y] = 0.0
                y[..., 2:3][mask_y] = 0.0
                mask_tensor = ~torch.isnan(y)
                y = torch.nan_to_num(y, nan=0.0)
            return x, y, mask_tensor
        pde_idx = idx // (self.nt - self.num_hist) # Markov / teacher forcing: only return one future step
        t_idx = idx % (self.nt - self.num_hist) + self.num_hist
        pde_path = self.data[pde_idx]
        # path_idx = os.path.join(path_root, str(pde_idx) + ".pt")
        # pde = torch.load(path_idx)
        pde = torch.load(pde_path)
        pde = pde.permute(1, 2, 0, 3)
        x = pde[..., (t_idx - self.num_hist):t_idx, :]
        mask = (x[..., 0:1] == 0.0)
        x[..., 1:2][mask] = 0.0
        x[..., 2:3][mask] = 0.0
        # mask_tensor = torch.isnan(x[..., :3])
        x[..., :3] = torch.nan_to_num(x[..., :3], nan=0.0)
        max_z = x[..., -1].max()
        z_value = max_z + 30.0
        x[..., -1] = torch.nan_to_num(x[..., -1], nan=z_value)
        x[..., -1] = torch.nn.functional.normalize(x[..., -1])
        x[..., 3:4] = self.log_transform(x[..., 3:4]) / 10.0
        # x[..., :4] = self.log_transform(x[..., :4])
        y = pde[..., t_idx, :3]
        mask_y = (y[..., 0:1] == 0.0)
        y[..., 1:2][mask_y] = 0.0
        y[..., 2:3][mask_y] = 0.0
        mask_tensor = ~torch.isnan(y)
        y = torch.nan_to_num(y, nan=0.0)
        if self.noise_std > 0:
            x += torch.randn(*x.shape, device=x.device) * self.noise_std

        return x, y, mask_tensor

################################################################
# Lploss: code from https://github.com/zongyi-li/fourier_neural_operator
################################################################
#loss function with rel/abs Lp loss
class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        # Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        # Assume uniform mesh
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
        assert x.shape == y.shape and len(x.shape) == 3, "wrong shape"
        # diff_norms = torch.norm(x.reshape(num_examples, -1) - y.reshape(num_examples, -1) + 1e-3, self.p, 1)
        # y_norms = torch.norm(y.reshape(num_examples, -1) + 1e-3, self.p, 1)
        diff_norms = torch.norm(x - y, self.p, 1)
        y_norms = torch.norm(y, self.p, 1)

        if self.reduction:
            loss = (diff_norms/(y_norms)).mean(-1) # average over channel dimension
            if self.size_average:
                return torch.mean(loss)
            else:
                return torch.sum(loss)

        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.rel(x, y)


def nse(y_pred, y_true):
    numerator = torch.sum((y_true - y_pred) ** 2)
    denominator = torch.sum((y_true - torch.mean(y_true)) ** 2)
    nse = 1.0 - numerator / denominator
    return nse

def corr(y_pred, y_true):
    mean_true = torch.mean(y_true)
    mean_pred = torch.mean(y_pred)
    cov = torch.mean((y_true - mean_true) * (y_pred - mean_pred))
    std_true = torch.std(y_true)
    std_pred = torch.std(y_pred)
    cor  = cov / (std_true * std_pred)
    return cor

def critical_success_index(y_pred, y_true, threshold):
    hits = torch.sum(torch.logical_and(y_true > threshold, y_pred > threshold))
    misses = torch.sum(torch.logical_and(y_true > threshold, y_pred <= threshold))
    false_alarms = torch.sum(torch.logical_and(y_true <= threshold, y_pred > threshold))
    csi = hits / (hits + misses + false_alarms + 1e-8)
    return csi

################################################################
# equivariance checks
################################################################
# function for checking equivariance to 90 rotations of a scalar field
def eq_check_rt(model, x, spatial_dims):
    model.eval()
    diffs = []
    with torch.no_grad():
        out = model(x)
        out[out == 0] = float("nan")
        for j in range(len(spatial_dims)):
            for l in range(j + 1, len(spatial_dims)):
                dims = [spatial_dims[j], spatial_dims[l]]
                diffs.append([((out.rot90(k=k, dims=dims) - model(x.rot90(k=k, dims=dims))) / out.rot90(k=k, dims=dims)).abs().nanmean().item() * 100 for k in range(1, 4)])
    return torch.tensor(diffs).mean().item()

# function for checking equivariance to reflections of a scalar field
def eq_check_rf(model, x, spatial_dims):
    model.eval()
    diffs = []
    with torch.no_grad():
        out = model(x)
        out[out == 0] = float("nan")
        for j in spatial_dims:
            diffs.append(((out.flip(dims=(j, )) - model(x.flip(dims=(j, )))) / out.flip(dims=(j, ))).abs().nanmean().item() * 100)
    return torch.tensor(diffs).mean().item()

################################################################
# grids
################################################################
class grid(torch.nn.Module):
    def __init__(self, twoD, grid_type):
        super(grid, self).__init__()
        assert grid_type in ["cartesian", "symmetric", "None"], "Invalid grid type"
        self.symmetric = grid_type == "symmetric"
        self.include_grid = grid_type != "None"
        self.grid_dim = (1 + (not self.symmetric) + (not twoD)) * self.include_grid
        if self.include_grid:
            if twoD:
                self.get_grid = self.twoD_grid
            else:
                self.get_grid = self.threeD_grid
        else:
            self.get_grid = torch.nn.Identity()
    def forward(self, x):
        return self.get_grid(x)

    def twoD_grid(self, x):
        shape = x.shape
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.linspace(0, 1, size_x).reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.linspace(0, 1, size_y).reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        if not self.symmetric:
            grid = torch.cat((gridx, gridy), dim=-1)
        else:
            midx = 0.5
            midy = (size_y - 1) / (2 * (size_x - 1))
            gridx = (gridx - midx) ** 2
            gridy = (gridy - midy) ** 2
            grid = gridx + gridy
        grid = grid.to(x.device)
        return torch.cat((x, grid), dim=-1)

    def threeD_grid(self, x):
        shape = x.shape
        batchsize, size_x, size_y, size_z = shape[0], shape[1], shape[2], shape[3]
        gridx = torch.linspace(0, 1, size_x).reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
        gridy = torch.linspace(0, 1, size_y).reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
        gridz = torch.linspace(0, 1, size_z).reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
        if not self.symmetric:
            grid = torch.cat((gridx, gridy, gridz), dim=-1)
        else:
            midx = 0.5
            midy = (size_y - 1) / (2 * (size_x - 1))
            gridx = (gridx - midx) ** 2
            gridy = (gridy - midy) ** 2
            grid = torch.cat((gridx + gridy, gridz), dim=-1)
        grid = grid.to(x.device)
        return torch.cat((x, grid), dim=-1)

################################################################
# fourier continuation for fft and ders on non-periodic data
################################################################
class fc(object):
    def __init__(self, fc_d=3, fc_C=25):
        super(fc, self).__init__()

        self.fc_d = fc_d
        self.fc_C = fc_C

        fc_data_pairs = ((3, 6), (3, 12), (3, 25), (5, 25))
        assert (fc_d, fc_C) in fc_data_pairs, ">> Error: precomputed FC matrices not found for the input degree!"
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.fc_d == 3 and self.fc_C == 25:
            fc_A = torch.tensor([[0.577350269189425, 1.41421356237007, 4.08248290457551],
                                 [0.577350268932529, 2.12132033966420, 10.2062071802068],
                                 [0.577350221019144, 2.82842639268168, 18.7794060152318],
                                 [0.577347652221430, 3.53549399024348, 29.8012851068571],
                                 [0.577291059875212, 4.24173355523443, 43.2551322460336],
                                 [0.576658953051786, 4.93909676130202, 58.9693320070330],
                                 [0.572570029004639, 5.58268162484116, 75.9760350811826],
                                 [0.555837186090934, 6.02703866329299, 91.0846940619291],
                                 [0.509831647972551, 6.00048596828142, 98.1277933706341],
                                 [0.421270599722172, 5.26159744521892, 91.0174238476298],
                                 [0.298559139312803, 3.88635368624552, 69.8896198069309],
                                 [0.174035832724397, 2.33081834487773, 43.0524050354295],
                                 [0.0806753357082983, 1.10206203421019, 20.7407332422353],
                                 [0.0288930418776341, 0.400289045332448, 7.63521737660355],
                                 [0.00777163554111367, 0.108783666634782, 2.09560532833085],
                                 [0.00152109005402703, 0.0214568816355638, 0.416455939791407],
                                 [0.000208380234960350, 0.00295706340634324, 0.0577289864413102],
                                 [1.89994028313830e-05, 0.000270889262028986, 0.00531296816181229],
                                 [1.07736236480021e-06, 1.54193275128022e-05, 0.000303559104610551],
                                 [3.45792118873408e-08, 4.96450646712166e-07, 9.80397286157291e-06],
                                 [5.48560209319221e-10, 7.89621720928448e-09, 1.56342203350098e-07],
                                 [3.50735040678574e-12, 5.05981661205677e-11, 1.00404742641513e-09],
                                 [6.52264500069445e-15, 9.42766487410956e-14, 1.87435770403075e-12],
                                 [1.99918217690639e-18, 2.89433588113721e-17, 5.76393682469331e-16],
                                 [3.27190167760519e-23, 4.74378753707539e-22, 9.46086295849135e-21]])
            fc_Q = torch.tensor([[0.577350269189626, -0.707106781186548, 0.408248290463863],
                                 [0.577350269189626, 0, -0.816496580927726],
                                 [0.577350269189626, 0.707106781186548, 0.408248290463863]])
        elif self.fc_d == 3 and self.fc_C == 6:
            fc_A = torch.tensor([[0.483868985307726, 0.964483282153122, 1.45399048686443],
                                 [0.327637155684739, 0.837213869787502, 1.88530013923995],
                                 [0.157342696770378, 0.457454312621708, 1.23309224217555],
                                 [0.0473720802649667, 0.148437703116213, 0.442738391766524],
                                 [0.00712731551515447, 0.0234174352696140, 0.0744836680937376],
                                 [0.000305680893875259, 0.00103738994246047, 0.00344990240574944]])
            fc_Q = torch.tensor([[0.577350269189626, -0.707106781186571, 0.408248290469261],
                                 [0.577350269189626, -2.35513868802566e-14, -0.816496580922371],
                                 [0.577350269189626, 0.707106781186524, 0.408248290469176]])
        elif self.fc_d == 3 and self.fc_C == 12:
            fc_A = torch.tensor([[0.573546998158633, 1.38271879584259, 3.56054049955439],
                                 [0.556560125288930, 1.94492807228321, 7.47317548872105],
                                 [0.510066046718126, 2.24265998438925, 10.1475854187773],
                                 [0.424298914594900, 2.16089107636261, 10.1986450056090],
                                 [0.308443695632726, 1.72956553670865, 7.85073632070237],
                                 [0.189194505600595, 1.13212135790947, 4.61894020810674],
                                 [0.0943851736050239, 0.591158561877336, 2.01548026040243],
                                 [0.0365633586637526, 0.236821168912075, 0.607959970913247],
                                 [0.0102748722786736, 0.0683057678956682, 0.107245988674933],
                                 [0.00187121305539315, 0.0127093245213994, 0.00528234785760654],
                                 [0.000178691074999025, 0.00123687092094845, -0.00113592686301733],
                                 [5.54545715050955e-06, 3.90864060407427e-05, -9.70937307549835e-05]])
            fc_Q = torch.tensor([[0.577350269189626, -0.707106781186571, 0.408248290469261],
                                 [0.577350269189626, -2.35513868802566e-14, -0.816496580922371],
                                 [0.577350269189626, 0.707106781186524, 0.408248290469176]])
        elif self.fc_d == 5 and self.fc_C == 25:
            fc_A = torch.tensor(
                [[0.447213595213893, 0.948683295425315, 1.87082866551978, 4.42718838564417, 15.0598750754598],
                 [0.447213566471397, 1.26491079655245, 3.74165453387796, 13.2815313368687, 65.2589232783266],
                 [0.447212450573684, 1.58112823059934, 6.14689495510080, 28.4591041307591, 179.978922289879],
                 [0.447191374730374, 1.89715982691718, 9.08465342229828, 51.5175937512142, 396.488186950784],
                 [0.446966918238617, 2.21128543843541, 12.5362266932715, 83.8049115386688, 756.975150786320],
                 [0.445479186233559, 2.51347442016254, 16.3914387996148, 125.512986200577, 1292.74214074293],
                 [0.438908561109676, 2.76710296179998, 20.2420970314981, 172.993074162615, 1978.10474822220],
                 [0.418619122985368, 2.88754975451463, 23.1220571972341, 215.119536388226, 2663.08991751941],
                 [0.373300556399696, 2.75856449291266, 23.6344830594980, 234.093637832804, 3070.54159967224],
                 [0.298335785994355, 2.31825947632122, 20.8522387775748, 215.990933668143, 2952.25128032015],
                 [0.205072820092728, 1.65156463965807, 15.3760499988086, 164.403004449191, 2313.95761273702],
                 [0.117049714956061, 0.966895368460470, 9.22502263196689, 100.901672765237, 1450.52467458190],
                 [0.0538541955574408, 0.453065603931456, 4.39999966589100, 48.9319254026378, 714.482617678240],
                 [0.0194453279835376, 0.165805429884090, 1.63157769459152, 18.3717245751179, 271.440908271919],
                 [0.00536208084274965, 0.0461885649130693, 0.459094234574135, 5.21916746800331, 77.8230392186869],
                 [0.00109492152815563, 0.00950637133551385, 0.0952341334581193, 1.09087064775007, 16.3851178230366],
                 [0.000159433272131088, 0.00139297290432927, 0.0140428463158209, 0.161840379168189, 2.44536200073336],
                 [1.57701275490821e-05, 0.000138492042265180, 0.00140339177869927, 0.0162554258102865,
                  0.246829722357186],
                 [9.93148306914283e-07, 8.75900860591491e-06, 8.91421685490032e-05, 0.00103691256056054,
                  0.0158107748841144],
                 [3.64187707221359e-08, 3.22354787675283e-07, 3.29273616098822e-06, 3.84404683743866e-05,
                  0.000588242397881924],
                 [6.84445515750482e-10, 6.07711557009970e-09, 6.22727773843944e-08, 7.29280794087538e-07,
                  1.11948617427213e-05],
                 [5.44798260931865e-12, 4.85034977964738e-11, 4.98403470395195e-10, 5.85298754658622e-09,
                  9.00944917393720e-08],
                 [1.35744246961945e-14, 1.21143860339078e-13, 1.24790124441774e-12, 1.46907623627354e-11,
                  2.26690195327967e-10],
                 [6.28970107833331e-18, 5.62526369280258e-17, 5.80740395643517e-16, 6.85182118735231e-15,
                  1.05963638253108e-13],
                 [1.96148342759244e-22, 1.75768510785428e-21, 1.81823606968664e-20, 2.14954096308993e-19,
                  3.33098142686937e-18]])
            fc_Q = torch.tensor(
                [[0.447213595499958, -0.632455532033676, 0.534522483824849, -0.316227766016838, 0.119522860933439],
                 [0.447213595499958, -0.316227766016838, -0.267261241912424, 0.632455532033676, -0.478091443733757],
                 [0.447213595499958, 0, -0.534522483824849, 0, 0.717137165600636],
                 [0.447213595499958, 0.316227766016838, -0.267261241912424, -0.632455532033676, -0.478091443733757],
                 [0.447213595499958, 0.632455532033676, 0.534522483824849, 0.316227766016838, 0.119522860933439]])

        fc_F1 = torch.flipud(torch.eye(self.fc_d))
        fc_F2 = torch.flipud(torch.eye(self.fc_C))

        self.fc_AQ = torch.matmul(fc_A, fc_Q.T).to(device)
        self.fc_AlQl = torch.matmul(torch.matmul(torch.matmul(fc_F2, fc_A), fc_Q.T), fc_F1).to(device)

    def fc_pad(self, x, domain_size=1.0):
        n_pts_x, n_pts_y, n_pts_z = x.shape[-3], x.shape[-2], x.shape[-1]
        # self.fc_A, self.fc_Q = self.fc_A.to(x.device), self.fc_Q.to(x.device)

        fc_h = domain_size / (n_pts_z - 1)
        fc_npoints_total = n_pts_z + self.fc_C
        fc_prd = fc_npoints_total * fc_h

        x = torch.cat((x, torch.matmul(self.fc_AlQl, x[..., :self.fc_d].unsqueeze(-1)).squeeze(-1)
                       + torch.matmul(self.fc_AQ, x[..., -self.fc_d:].unsqueeze(-1)).squeeze(-1)), dim=-1)

        return x, fc_prd

    def __call__(self, x, domain_size=1.0):
        return self.fc_pad(x, domain_size)