import torch
from torch.nn import Module
from typing import Callable

# YOU NEED TO MAKE A PARTICLE CLASS BECUASE YOU NEED TO HOLD A VELOCTY AND POSITION VECTOR ALONG WITH A LOSS AND ACCURACY VALUE!!!!

APLHA = 0.5
BETA = 0.01
M = 100

class PSO():

    def __init__(self, num_particles: int, num_dimensions: int, net: Module, device: str):
        self.num_particles = num_particles
        self.num_dimensions = num_dimensions
        self.device = device
        self.net = net

        i = torch.linspace(0, self.num_particles-2, 1)
        self.learning_probabilities = torch.pow((1-(i/self.num_particles)), APLHA*torch.log(torch.ceil(self.num_particles/M)))

        # Create particles
        self.particles = torch.rand((self.num_particles, self.num_dimensions)).to(self.device)

    def evalute(self, particle: torch.Tensor, closure: Callable):
        net_params = self.net.parameters()

        # Copy parameters into network
        param_pointer = 0
        for param in net_params:
            param_len = torch.prod(torch.tensor(param.shape))
            p_param = particle[param_pointer:param_pointer+param_len]
            p_param = p_param.reshape(param.shape)
            param.data = p_param

            param_pointer += param_len

        loss, accuracy = closure()

        return loss, accuracy

    def update_sl(self, losses: torch.Tensor):
        idxs = torch.argsort(losses, descending=True)
        self.particles = torch.gather(self.particles, idxs)

        p = torch.rand(self.num_particles)
        does_learn = torch.le(p, self.learning_probabilities)
        for i, particle, learn in enumerate(zip(self.particles[:-1], does_learn)):
            if learn:
                teacher = torch.randint(i+1, len(self.particles)-1)



    def step(self, closure: Callable):
        # Choose a redundant best with higest possible score to make sure they are beat
        self.global_best = self.particles[0].clone().detach()
        self.global_best_score = torch.inf

        scores = torch.empty((self.num_particles, 2))
        for particle, score in zip(self.particles, scores):
            loss, accuracy = self.evalute(particle, closure)

            if loss < self.global_best_score:
                self.global_best = particle.clone().detach()
                self.global_best_score = loss

            score[0] = loss
            score[1] = accuracy

        


        
