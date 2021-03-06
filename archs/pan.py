import torch
from torch import nn


class PAN(nn.Module):
    def __init__(self, input_size):
        super(PAN, self).__init__()
        self.fc1 = nn.Linear(input_size, 80)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(80, 60)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(60, 2)

    def forward(self, data, out_feature=False):
        output = self.fc1(data)
        output = self.relu1(output)
        output = self.fc2(output)
        output = self.relu2(output)
        output = self.fc3(output)
        return output


class AgnosticPAN(nn.Module):
    def __init__(self, input_size):
        super(AgnosticPAN, self).__init__()
        self.fc1 = nn.Linear(input_size, 80)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(80, 50)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(50, 20)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(20, 2)

    def forward(self, data, out_feature=False):
        output = self.fc1(data)
        output = self.relu1(output)
        output = self.fc2(output)
        output = self.relu2(output)
        output = self.fc3(output)
        output = self.relu3(output)
        output = self.fc4(output)
        return output


def compute_agnostic_stats(data):
    mean = torch.mean(data, dim=1)
    std = torch.std(data, dim=1)
    maximum = torch.max(data, dim=1)[0]

    # def percentile(t: torch.tensor, q: float):
    #     k = 1 + round(0.01 * float(q) * (t.numel() - 1))
    #     result = t.view(-1).kthvalue(k).values.item()
    #     return result

    # third_q = []
    # for n in range(len(data)):
    #     third_q.append(percentile(data[n], 75))
    # third_q = torch.tensor(third_q).to("cuda")

    minimum = torch.min(data, dim=1)[0]
    median = torch.median(data, dim=1)[0]
    # range1 = maximum-minimum
    # range2 = maximum-median

    res = torch.stack([mean, std, maximum, minimum, median], dim=1)
    return res
