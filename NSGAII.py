from torch.nn import Module, Parameter
from torch.optim import Optimizer
import torch
from typing import Iterator, List, Callable, Dict
from random import randint
import operator

class Induvidual():

    def __init__(self, device, parameters: List[Parameter], min_val: int, max_val: int, starter: List[Parameter] = None):
        self.device = device
        self.master = parameters
        self.score = None
        self.min = min_val
        self.max = max_val
        self.valid = False
        self.crowding_distance = 0

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
            self.score = closure()
            self.valid = True
            self.crowding_distance = 0

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

        return Induvidual(self.device, self.master, self.min, self.max, child1), Induvidual(self.device, self.master, self.min, self.max, child2)

class NSGA():

    def __init__(self, parameters: Iterator[Parameter], num_induviduals: int, device, min_val: int = -10, max_val: int = 10):
        self.device = device
        self.master = []
        for param in parameters:
            self.master.append(param)

        self.num_induviduals = num_induviduals 
        self.min_val = min_val
        self.max_val = max_val

        self.induviduals = [Induvidual(self.device, self.master, self.min_val, self.max_val) for _ in range(self.num_induviduals)]

    def dominated(self, front: List[Induvidual], ind: Induvidual):
        for f_ind in front:
            # A>B . C<=D + C<D . A>=B
            if((f_ind.score[0] > ind.score[0] or f_ind.score[1] < ind.score[1])
                and
                (f_ind.score[0] >= ind.score[0] and f_ind.score[1] <= ind.score[1])):
                return True
        return False

    def crowding_distance(self, front: List[Induvidual], max_scores: List[int], min_scores: List[int]):
        for i in range(len(front[0].score)):
            sorted_front = sorted(front, key=operator.attrgetter('score')[i])
            sorted_front[0].crowding_distance += torch.inf
            sorted_front[-1].crowding_distance += torch.inf

            for j in range(1, len(front)-1):
                d = torch.abs(sorted_front[j-1].score[i]-sorted_front[j+1].score[i])
                dHat = d/(max_scores[i]-min_scores[i])
                sorted_front[j].crowding_distance += dHat

        front.sort(key=operator.attrgetter('crowding_distance'), reverse=True)

        return front

    def get_fronts(self):
        self.induviduals.sort(key=operator.attrgetter('score')[1])

        fronts: Dict[int, List[Induvidual]] = {0: []}

        for ind in self.induviduals:
            dom = True
            for front in fronts.values():
                dom = self.dominated(front, ind)
                if not dom:
                    front.append(ind)
                    break
            if dom:
                fronts[fronts.keys()[-1]+1] = [ind]
            

    def selection(self, max_scores, min_scores):
        fronts: Dict[int, List[Induvidual]] = self.get_fronts()

        new_ind = []
        
        current_front = 0
        while(len(new_ind)+len(fronts[current_front]) <= self.num_induviduals):
            for ind in fronts[current_front]:
                new_ind.append(ind)
            current_front+=1

        if(len(new_ind) != self.num_induviduals):
            for ind in fronts[current_front]:
                ind.crowding_distance = 0

            sorted_front = self.crowding_distance(fronts[current_front], max_scores, min_scores)

            for i in range(self.num_induviduals-len(new_ind)):
                new_ind.append(sorted_front[i])

        return new_ind

    def step(self, closure: Callable):
        max_scores = [0]*len(self.induviduals[0].score)
        min_scores = [torch.inf]*len(self.induviduals[0].score)

        for induvidual in self.induviduals:
            induvidual.evaluate(closure)

            for i in range(len(max_scores)):
                max_scores[i] = max(max_scores[i], induvidual.score[i])
                min_scores[i] = min(min_scores[i], induvidual.score[i])
            
        self.induviduals: List[Induvidual] = self.selection(max_scores, min_scores)

        # Should load one of the best induviduals from the first front into the network for validation and testing
        self.induviduals[0].load_params()

        return self.induviduals[0].score

