from torch.nn import Module, Parameter
from torch.optim import Optimizer
import torch
from typing import Iterator, List, Callable
from random import randint
import operator

class Induvidual():

    def __init__(self, parameters: List[Parameter], min_val: int, max_val: int, starter: List[Parameter] = None):
        self.master = parameters
        self.accuracy = 0
        self.complexity = 0
        self.min = min_val
        self.max = max_val
        self.valid = False
        self.device = parameters[0].get_device()
        self.front = 0

        if starter is not None:
            self.params = starter
        else:
            self.params: List[Parameter] = []
            for param in parameters:
                self.params.append(min_val+(max_val-min_val)*torch.rand(param.shape).to(self.device))

    def load_params(self):
        for mp, ip in zip(self.master, self.params):
            mp.data = ip.data

    def evaluate(self, closure: Callable):
        if(not self.valid):
            self.load_params()
            self.accuracy, self.complexity = closure()
            self.valid = True

    def mutate(self):
        param_idx = randint(0, len(self.params)-1)
        self.params[param_idx].data = self.min_val+(self.max_val-self.min_val)*torch.rand(self.params[param_idx].shape).to(self.device)
        self.valid = False

    def crossover(self, mate: List[Parameter]):
        feature_inheritence = torch.randint(low=0, high=1, size=len(self.params)).to(self.device)

        child1 = []
        child2 = []
        for fe, mp, ip in zip(feature_inheritence, mate, self.params):
            if(fe):
                child1.append(ip.data.detach().clone())
                child2.append(mp.data.detach().clone())
            else:
                child1.append(mp.data.detach().clone())
                child2.append(ip.data.detach().clone())

        return Induvidual(self.master, self.min, self.max, child1), Induvidual(self.master, self.min, self.max, child2)

class NSGA():

    def __init__(self, parameters: Iterator[Parameter], num_induviduals: int, min_val: int = -10, max_val: int = 10):
        self.master = []
        for param in parameters:
            self.master.append(param)

        self.num_induviduals = num_induviduals 
        self.min_val = min_val
        self.max_val = max_val

        self.induviduals = [Induvidual(self.master, self.min_val, self.max_val) for _ in range(self.num_induviduals)]

    def dominated(self, front: List[Induvidual], ind: Induvidual):
        for f_ind in front:
            # A>B . C<=D + C<D . A>=B
            if((f_ind.accuracy > ind.accuracy or f_ind.complexity < ind.complexity)
                and
                (f_ind.accuracy >= ind.accuracy and f_ind.complexity <= ind.complexity)):
                return True
        return False

    def ENDS(self):
        #sorts from lowest complexity (good) to highest (bad)
        self.induviduals.sort(key=operator.attrgetter('complexity'))

        fronts = {0: []}

        for ind in self.induviduals:
            dom = True
            for front in fronts.values():
                dom = self.dominated(front, ind)
                if not dom:
                    front.append(ind)
                    break
            if dom:
                fronts[fronts.keys()[-1]+1] = [ind]

        # Move fronts into induviduals then use crowding distance
                


    def step(self, closure: Callable):
        for induvidual in self.induviduals:
            induvidual.evaluate(closure)


        


        

