import torch
import torch.nn as nn
import torch.nn.functional as F

class Gate(nn.Module):
    def __init__(self, num, input_dim):
        super().__init__()
        self.num = num
        self.w = nn.ModuleList([nn.Linear(input_dim,1, bias=False) for _ in range(num)])
    def forward(self, x, experts):
        weight = torch.cat([w(x) for w in self.w], 1)
        weigth = F.softmax(weight,1)
        return torch.bmm(weigth.unsqueeze(1), experts).squeeze()

class CGC(nn.Module):
    def __init__(self, input_dim, expert_dim, expert_num):
        super(CGC, self).__init__()
        self.expert_a = nn.ModuleList([nn.Linear(input_dim, expert_dim) for _ in range(expert_num)])
        self.expert_b = nn.ModuleList([nn.Linear(input_dim, expert_dim) for _ in range(expert_num)])
        self.expert_share = nn.ModuleList(nn.Linear(input_dim, expert_dim) for _ in range(expert_num))
        self.ta = nn.Linear(expert_dim, 1)
        self.tb = nn.Linear(expert_dim, 1)
        self.gate_a = Gate(expert_num * 2, input_dim)
        self.gate_b = Gate(expert_num * 2, input_dim)
    def forward(self, x):
        expert_a = torch.cat([e(x) for e in self.expert_a], 1)
        expert_b = torch.cat([e(x) for e in self.expert_b], 1)
        expert_share = torch.cat([e(x) for e in self.expert_share], 1)
        expert_a = self.gate_a(x, torch.cat([expert_a, expert_share], 1))
        expert_b = self.gate_b(x, torch.cat([expert_b, expert_share], 1))
        y_a = self.ta(expert_a)
        y_b = self.tb(expert_b)
        return y_a, y_b

class PLElayers(nn.Module):
    def __init__(self, input_dim, expert_dim, expert_num):
        super(PLElayers, self).__init__()
        self.expert_a = nn.ModuleList([nn.Linear(input_dim, expert_dim) for _ in range(expert_num)])
        self.expert_b = nn.ModuleList([nn.Linear(input_dim, expert_dim) for _ in range(expert_num)])
        self.expert_share = nn.ModuleList(nn.Linear(input_dim, expert_dim) for _ in range(expert_num))
        self.gate_a = Gate(expert_num * 2, input_dim)
        self.gate_b = Gate(expert_num * 2, input_dim)
        self.gate_share = Gate(expert_num * 3, input_dim)
    def forward(self, x):
        x_a, x_b, x_share = x
        expert_a = torch.cat([e(x_a).unsqueeze(1) for e in self.expert_a], 1)
        expert_b = torch.cat([e(x_b).unsqueeze(1) for e in self.expert_b], 1)
        expert_share = torch.cat([e(x_share).unsqueeze(1) for e in self.expert_share], 1)
        expert_a_out = self.gate_a(x_a, torch.cat([expert_a, expert_share], 1))
        expert_b_out = self.gate_b(x_b, torch.cat([expert_b, expert_share], 1))
        expert_share_out = self.gate_share(x_share, torch.cat([expert_a, expert_b, expert_share], 1))
        return expert_a_out, expert_b_out, expert_share_out

class PLE(nn.Module):
    def __init__(self, input_dim, expert_dim, expert_num, layers):
        super(PLE, self).__init__()
        self.cgc = [PLElayers(input_dim, expert_dim, expert_num)]
        for _ in range(1, layers):
            self.cgc.append(PLElayers(expert_dim, expert_dim, expert_num))
        self.cgc = nn.Sequential(*self.cgc)
        self.ta = nn.Linear(expert_dim, 1)
        self.tb = nn.Linear(expert_dim, 1)
    def forward(self, x):
        expert_a, expert_b, expert_share = self.cgc((x,x,x))
        y_a = self.ta(expert_a)
        y_b = self.tb(expert_b)
        return y_a, y_b

if __name__ == '__main__':

    ple = PLE(4, 4, 2, 2)
    x = torch.rand((3, 4))
    y_a, y_b = ple(x)
    print(y_a.shape, y_b.shape)
