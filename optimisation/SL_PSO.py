import torch
from torch.nn import Module
from typing import Callable
import random
import operator
from model_info import get_param_count
import numpy as np

APLHA = 0.5
BETA = 0.01
M = 100

class Particle():

    def __init__(self, num_dimensions, device, id):
        self.id = id
        self.num_dimensions = num_dimensions
        self.device = device

        self.position = -10+20*torch.rand(self.num_dimensions).to(device)
        self.velocity = -10+20*torch.rand(self.num_dimensions).to(device)

        self.loss = 0
        self.accuracy = 0

    def evaluate(self, closure: Callable):
        self.loss, self.accuracy = closure(self.id)

    def learn_from(self, teacher, r, epsilon, mean_dims):
        self.velocity = r[0]*self.velocity + r[1]*(teacher.position-self.position) + r[2]*epsilon*(mean_dims-self.position)
        self.position += self.velocity
        

class PSO():

    def __init__(self, net: Module, device: str, num_particles: int = None):
        self.device = device
        self.net = net

        self.num_dimensions = get_param_count(self.net)

        if num_particles is None:
            self.num_particles = int(M+np.floor(self.num_dimensions/10))
        else:
            self.num_particles = num_particles

        print(self.num_particles)

        self.epsilon = BETA*(self.num_dimensions/M)

        i = torch.linspace(0, self.num_particles-2, 1).to(device)
        self.learning_probabilities = torch.pow((1-(i/self.num_particles)), APLHA*torch.log(torch.ceil(torch.tensor([self.num_particles/M]).to(device))))

        # Create particles
        self.particles = [Particle(self.num_dimensions, self.device, i) for i in range(self.num_particles)]

    def load_net_params(self, particle: Particle):
        param_pointer = 0

        for param in self.net.parameters():
            param_len = torch.prod(torch.tensor(param.shape))
            p_param = particle.position[param_pointer:param_pointer+param_len]
            p_param = p_param.reshape(param.shape)
            param.data = p_param

            param_pointer += param_len

    def evalute(self, particle: Particle, closure: Callable):
        self.load_net_params(particle)

        particle.evaluate(closure)

    def calc_mean(self):
        result = torch.zeros(self.num_dimensions).to(self.device)

        for particle in self.particles:
            result += particle.position

        result /= self.num_particles

        return result

    def update_sl(self, mean_dims):
        self.particles.sort(key=operator.attrgetter('loss'), reverse=True)

        p = torch.rand(self.num_particles).to(self.device)
        does_learn = torch.le(p, self.learning_probabilities)

        acceleration = torch.rand((self.num_particles, 3))

        for i, (learner, learn, r_set) in enumerate(zip(self.particles[:-1], does_learn, acceleration)):
            if learn:
                teacher = self.particles[random.randint(i+1, len(self.particles)-1)]
                learner.learn_from(teacher, r_set, self.epsilon, mean_dims)

    def step(self, closure: Callable):

        for particle in self.particles:
            self.evalute(particle, closure)

        mean_dims = self.calc_mean()

        self.update_sl(mean_dims)

        self.load_net_params(self.particles[-1])

        return self.particles[-1].loss, self.particles[-1].accuracy

