# There are random function from 8 inputs and X random inputs added.
# We split data in 2 parts, on first part you will train and on second
# part we will test
import time
import random
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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

class SolutionModel(nn.Module):
    def __init__(self, input_size, output_size, solution):
        super(SolutionModel, self).__init__()
        sizes = [input_size] + solution.hidden_sizes + [output_size]
        sizes_pairs = list(zip(sizes, sizes[1:]))
        sub_models = [[
            BatchInitLinear(a, b, solution),
            nn.Sigmoid() if ind == len(sizes_pairs)-1 else nn.ReLU()
            ] for ind, (a, b) in enumerate(sizes_pairs)]
        sub_models_flat = sum(sub_models, [])
        self.seq = nn.Sequential(*sub_models_flat)
        self.loss = nn.BCELoss()

    def forward(self, x):
        x = (x-0.5)*2.0
        return self.seq(x)

    def calc_error(self, output, target):
        return self.loss(output, target)

    def calc_predict(self, output):
        return output.round()

class Solution():
    def __init__(self):
        self.init_type = 'uniform'
        self.learning_rate = 0.5
        self.momentum = 0.0
        self.layers_number = 3
        self.first_hidden_size = 16
        self.hidden_size = 32
        self.batch_size = 128

    # Return trained model
    def train_model(self, train_data, train_target, context):
        self.hidden_sizes = [self.first_hidden_size] + [self.hidden_size]*(self.layers_number-1)
        model = SolutionModel(train_data.size(1), train_target.size(1), self)
        batches = train_data.size(0)//self.batch_size
        good_count = 0
        good_limit = batches
        optimizer = optim.SGD(model.parameters(), lr=self.learning_rate, momentum=self.momentum)

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

    def create_data(self, data_size, input_size, random_input_size, seed):
        torch.manual_seed(seed)
        function_size = 1 << input_size
        function_input = torch.ByteTensor(function_size, input_size)
        for i in range(function_input.size(0)):
            fun_ind = i
            for j in range(function_input.size(1)):
                input_bit = fun_ind&1
                fun_ind = fun_ind >> 1
                function_input[i][j] = input_bit
        function_output = torch.ByteTensor(function_size).random_(0, 2)

        if data_size % function_size != 0:
            raise "Data gen error"

        data_input = torch.ByteTensor(data_size, input_size).view(-1, function_size, input_size)
        target = torch.ByteTensor(data_size).view(-1, function_size)
        for i in range(data_input.size(0)):
            data_input[i] = function_input
            target[i] = function_output
        data_input = data_input.view(data_size, input_size)
        target = target.view(data_size)
        if random_input_size > 0:
            data_random = torch.ByteTensor(data_size, random_input_size).random_(0, 2)
            data = torch.cat([data_input, data_random], dim=1)
        else:
            data = data_input
        perm = torch.randperm(data.size(1))
        data = data[:,perm]
        perm = torch.randperm(data.size(0))
        data = data[perm]
        target = target[perm]
        return (data.float(), target.view(-1, 1).float())

    def create_case_data(self, case):
        data_size = 256*32
        input_size = 8
        random_input_size = min(32, (case-1)*4)

        data, target = self.create_data(2*data_size, input_size, random_input_size, case)
        return sm.CaseData(case, Limits(), (data[:data_size], target[:data_size]), (data[data_size:], target[data_size:])).set_description("{} inputs and {} random inputs".format(input_size, random_input_size))

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
