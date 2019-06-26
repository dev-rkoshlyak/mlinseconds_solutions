# HelloXor is a HelloWorld of Machine Learning.
import time
import random
import argparse
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ..utils import solutionmanager as sm
from ..utils import gridsearch as gs

class SolutionModel(nn.Module):
    def __init__(self, input_size, output_size, solution):
        super(SolutionModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = solution.hidden_size
        self.linear1 = nn.Linear(input_size, self.hidden_size)
        self.linear2 = nn.Linear(self.hidden_size, output_size)

    def forward(self, x):
        x = self.linear1(x)
        x = torch.sigmoid(x)
        x = self.linear2(x)
        x = torch.sigmoid(x)
        return x

    def calc_error(self, output, target):
        # This is loss function
        return ((output-target)**2).sum()

    def calc_predict(self, output):
        # Simple round output to predict value
        return output.round()

class Solution():
    def __init__(self):
        # Control speed of learning
        self.learning_rate = 0.00001
        # Control number of hidden neurons
        self.hidden_size = 1
        # Maximum number of steps
        self.max_steps = 100
        # grid search will initialize this field
        self.grid_search = None
        # grid search will initialize this field
        self.iter = 0

    # Return trained model
    def train_model(self, train_data, train_target, context):
        # Model represent our neural network
        model = SolutionModel(train_data.size(1), train_target.size(1), self)
        # Optimizer used for training neural network
        optimizer = optim.SGD(model.parameters(), lr=self.learning_rate)
        step_till_correct = self.max_steps
        while True:
            # Report step, so we know how many steps
            context.increase_step()
            # model.parameters()...gradient set to zero
            optimizer.zero_grad()
            # evaluate model => model.forward(data)
            output = model(train_data)
            # if x < 0.5 predict 0 else predict 1
            predict = model.calc_predict(output)
            # Number of correct predictions
            correct = predict.eq(train_target.view_as(predict)).long().sum().item()
            # Total number of needed predictions
            total = predict.view(-1).size(0)
            # No more time left or learned everything, stop training
            time_left = context.get_timer().get_time_left()
            if total == correct:
                step_till_correct = min(step_till_correct, context.step)
            if context.step == self.max_steps:
                break
            # calculate error
            error = model.calc_error(output, train_target)
            # calculate deriviative of model.forward() and put it in model.parameters()...gradient
            error.backward()
            # print progress of the learning
            #self.print_stats(context.step, error, correct, total)
            # update model: model.parameters() -= lr * gradient
            optimizer.step()

        if self.grid_search:
            self.grid_search.add_result('error', error.item())
            self.grid_search.add_result('step', step_till_correct)
            if self.iter == self.iter_number-1:
                if self.iter == self.iter_number-1:
                    stats_error = self.grid_search.get_stats('error')
                    stats_step = self.grid_search.get_stats('step')
                    print('{} => Error {} {} Step {} {}'.format(self.grid_search.choice_str, *stats_error, *stats_step))
        return model

    def print_stats(self, step, error, correct, total):
        if step % 1000 == 0:
            print("Step = {} Correct = {}/{} Error = {}".format(step, correct, total, error.item()))

###
###
### Don't change code after this line
###
###
class Limits:
    def __init__(self):
        self.time_limit = 2.0
        self.size_limit = 100
        self.test_limit = 1.0

class DataProvider:
    def __init__(self):
        self.number_of_cases = 10

    def create_data(self):
        data = torch.FloatTensor([
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0]
            ])
        target = torch.FloatTensor([
            [0.0],
            [1.0],
            [1.0],
            [0.0]
            ])
        return (data, target)

    def create_case_data(self, case):
        data, target = self.create_data()
        return sm.CaseData(case, Limits(), (data, target), (data, target))

class Config:
    def __init__(self):
        self.max_samples = 10000

    def get_data_provider(self):
        return DataProvider()

    def get_solution(self):
        return Solution()

run_grid_search = False
# Uncomment next line if you want to run grid search
run_grid_search = True
if run_grid_search:
    parser = argparse.ArgumentParser(description='Grid search', allow_abbrev=False)
    parser.add_argument('-iter_number', type=int, default=10)
    parser.add_argument('-max_steps', type=int, nargs='+', default=[100])
    parser.add_argument('-hidden_size', type=int, nargs='+', default=[1])
    parser.add_argument('-learning_rate', type=float, nargs='+', default=[1.0])
    results_file = 'helloxor_day1_results_data.pickle'
    results_data = gs.ResultsData.load(results_file)
    gs.GridSearch().run(Config(), case_number=1, results_data=results_data, grid_parser=parser)
    results_data.save(results_file)
else:
    # If you want to run specific case, put number here
    sm.SolutionManager().run(Config(), case_number=-1)