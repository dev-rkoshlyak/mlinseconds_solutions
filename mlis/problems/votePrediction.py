# There are random function from 8 inputs.
# There are random input vector of size 8 * number of voters.
# We calculate function number of voters times and sum result.
# We return 1 if sum > voters/2, 0 otherwise
# We split data in 2 parts, on first part you will train and on second
# part we will test
import time
import random
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from ..utils import solutionmanager as sm
from ..utils import gridsearch as gs

class BatchInitLinear(nn.Linear):
    def __init__(self, fromSize, toSize, solution):
        super(BatchInitLinear, self).__init__(fromSize, toSize)
        self.solution = solution
        if solution.init_type == 'uniform':
            nn.init.uniform_(self.weight, a=-1.0, b=1.0)
            nn.init.uniform_(self.bias, a=-1.0, b=1.0)
        elif solution.init_type == 'normal':
            nn.init.normal_(self.weight, 0.0, 1.0)
            nn.init.normal_(self.bias, 0.0, 1.0)
        else:
            raise "Error"
        nn.init.constant_(self.bias, 0.0)
        self.first_run = True

    def forward(self, x):
        if not self.first_run:
            return super(BatchInitLinear, self).forward(x)
        else:
            self.first_run = False
            res = super(BatchInitLinear, self).forward(x)
            res_std = res.data.std(dim=0)
            self.weight.data /= res_std.view(res_std.size(0), 1).expand_as(self.weight)
            res.data /= res_std
            if self.bias is not None:
                self.bias.data /= res_std
                res_mean = res.data.mean(dim=0)
                self.bias.data -= res_mean
                res.data -= res_mean

        return res

class BaseModel(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, solution):
        super(BaseModel, self).__init__()
        sizes = [input_size] + hidden_sizes + [output_size]
        sizes_pairs = list(zip(sizes, sizes[1:]))
        sub_models = [[
            BatchInitLinear(a, b, solution),
            nn.Sigmoid() if ind == len(sizes_pairs)-1 else nn.ReLU()
            ] for ind, (a, b) in enumerate(sizes_pairs)]
        sub_models_flat = sum(sub_models, [])
        self.seq = nn.Sequential(*sub_models_flat)

    def forward(self, x):
        return self.seq(x)

class CompareModel(BaseModel):
    def __init__(self, input_size, output_size, solution):
        super(CompareModel, self).__init__(input_size, solution.compare_hidden_sizes, output_size, solution)

class RandFunctionEmbModel(nn.Module):
    def __init__(self, solution):
        super(RandFunctionEmbModel, self).__init__()
        self.voter_input = solution.voter_input
        self.emb = nn.Embedding(1<<solution.voter_input, solution.signal_count)

    def forward(self, x):
        x = x.view(-1, self.voter_input)
        x = x.byte().numpy()
        x = np.packbits(x, axis=-1)
        x = torch.from_numpy(x).long()
        x = self.emb(x)
        return x

class CompareSumModel(nn.Module):
    def __init__(self, voter_count, output_size, solution):
        super(CompareSumModel, self).__init__()
        self.voter_count = voter_count
        self.signal_count = solution.signal_count
        self.compare = CompareModel(self.signal_count, output_size, solution)

    def forward(self, x):
        x = x.view(x.size(0)//self.voter_count, self.voter_count, self.signal_count)
        x = x.sum(dim=1)
        x = x.view(x.size(0), -1)
        x = self.compare(x)
        return x

class VotePredictionModel(nn.Module):
    def __init__(self, input_size, output_size, solution):
        super(VotePredictionModel, self).__init__()
        self.voter_count = input_size // solution.voter_input
        self.rand_function = RandFunctionEmbModel(solution)
        self.compare = CompareSumModel(self.voter_count, output_size, solution)
        self.loss = nn.BCELoss()

    def forward(self, x):
        x = self.rand_function(x)
        x = self.compare(x)
        return x

    def calc_error(self, output, target):
        return self.loss(output, target)

    def calc_predict(self, output):
        return output.round()

class Solution():
    def __init__(self):
        self.learning_rate = 0.1
        self.voter_input = 8
        self.signal_count = 3
        self.batch_size = 128
        self.init_type = 'uniform'
        self.compare_hidden_sizes = []

    def create_model(self, input_size, output_size):
        return VotePredictionModel(input_size, output_size, self)

    def train_model(self, train_data, train_target, context):
        model = self.create_model(train_data.size(1), train_target.size(1))
        batches = train_data.size(0)//self.batch_size
        good_count = 0
        good_limit = 16
        optimizer = optim.RMSprop(model.parameters(), lr=self.learning_rate)

        while True:
            ind = context.step%batches
            context.increase_step()
            start_ind = self.batch_size*ind
            end_ind = start_ind + self.batch_size
            data = train_data[start_ind:end_ind]
            target = train_target[start_ind:end_ind]
            optimizer.zero_grad()
            output = model(data)
            with torch.no_grad():
                diff = (output-target).abs()
                if diff.max() <  0.3:
                    good_count += 1
                    if good_count >= good_limit:
                        break
                else:
                    good_count = 0
            error = model.calc_error(output, target)
            error.backward()
            optimizer.step()
            with torch.no_grad():
                predict = model.calc_predict(output)
                correct = predict.eq(target.view_as(predict)).long().sum().item()
                total = predict.view(-1).size(0)
            time_left = context.get_timer().get_time_left()
            if time_left < 0.1:
                break
        return model

###
###
### Don't change code after this line
###
###
class Limits:
    def __init__(self):
        self.time_limit = 2.0
        self.size_limit = 1000000
        self.test_limit = 1.0

class DataProvider:
    def __init__(self):
        self.number_of_cases = 10

    def get_index(self, tensor_index):
        index = 0
        for i in range(tensor_index.size(0)):
            index = 2*index + tensor_index[i].item()
        return index

    def calc_value(self, input_data, function_table, input_size, input_count_size):
        count = 0
        for i in range(input_count_size):
            count += function_table[self.get_index(input_data[i*input_size: (i+1)*input_size])].item()
        if count > input_count_size/2:
            return 1
        else:
            return 0

    def create_data(self, data_size, input_size, input_count_size, seed):
        torch.manual_seed(seed)
        function_size = 1 << input_size
        function_table = torch.ByteTensor(function_size).random_(0, 2)
        total_input_size = input_size*input_count_size
        data = torch.ByteTensor(data_size, total_input_size).random_(0, 2)
        target = torch.ByteTensor(data_size)
        for i in range(data_size):
            target[i] = self.calc_value(data[i], function_table, input_size, input_count_size)
        return (data.float(), target.view(-1, 1).float())

    def create_case_data(self, case):
        input_size = 8
        data_size = (1<<input_size)*32
        input_count_size = case

        data, target = self.create_data(2*data_size, input_size, input_count_size, case)
        return sm.CaseData(case, Limits(), (data[:data_size], target[:data_size]), (data[data_size:], target[data_size:])).set_description("{} inputs per voter and {} voters".format(input_size, input_count_size))

class Config:
    def __init__(self):
        self.max_samples = 10000

    def get_data_provider(self):
        return DataProvider()

    def get_solution(self):
        return Solution()

run_grid_search = False
# Uncomment next line if you want to run grid search
#run_grid_search = True
if run_grid_search:
    gs.GridSearch().run(Config(), case_number=1, random_order=False, verbose=False)
else:
    # If you want to run specific case, put number here
    sm.SolutionManager().run(Config(), case_number=-1)
