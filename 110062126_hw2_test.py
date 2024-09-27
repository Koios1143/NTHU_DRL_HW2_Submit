import torch
from torch import nn
from torchvision import transforms as T

import gym
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT

import numpy as np
import os, time
from collections import deque

class NoisyNetLayer(nn.Module):
    """
    NoisyNet layer, factorized version
    """
    def __init__(self, in_features, out_features, sigma=0.5):
        super().__init__()

        self.sigma = sigma
        self.in_features = in_features
        self.out_features = out_features

        # mu_b, mu_w, sigma_b, sigma_w
        # size: q, qxp, q, qxp
        self.mu_bias = nn.Parameter(torch.zeros(out_features))
        self.mu_weight = nn.Parameter(torch.zeros(out_features, in_features))
        self.sigma_bias = nn.Parameter(torch.zeros(out_features))
        self.sigma_weight = nn.Parameter(torch.zeros(out_features, in_features))

        self.cached_bias = None
        self.cached_weight = None

        self.register_noise_buffers()
        self.parameter_initialization()
        self.sample_noise()

    def forward(self, x, sample_noise=True):
        return nn.functional.linear(x, weight=self.mu_weight, bias=self.mu_bias)

    def register_noise_buffers(self):
        """
        Register noise f(epsilon_in) and f(epsilon_out)
        """
        self.register_buffer(name='epsilon_input', tensor=torch.empty(self.in_features))
        self.register_buffer(name='epsilon_output', tensor=torch.empty(self.out_features))

    def _calculate_bound(self):
        return self.in_features**(-0.5)

    @property
    def weight(self):
        """
        w = sigma \circ epsilon + mu
        epsilon = f(epsilon_in)f(epsilon_out)
        """
        if self.cached_weight is None:
            self.cached_weight = self.sigma_weight * torch.ger(self.epsilon_output, self.epsilon_input) + self.mu_weight
        return self.cached_weight

    @property
    def bias(self):
        """
        b = sigma \circ epsilon + mu
        """
        if self.cached_bias is None:
            self.cached_bias = self.sigma_bias * self.epsilon_output + self.mu_bias
        return self.cached_bias

    def sample_noise(self):
        """
        Sample factorised noise
        f(x) = sgn(x)\sqrt{|x|}
        """
        with torch.no_grad():
            epsilon_input = torch.randn(self.in_features, device=self.epsilon_input.device)
            epsilon_output = torch.randn(self.out_features, device=self.epsilon_output.device)
            self.epsilon_input = (epsilon_input.sign() * torch.sqrt(torch.abs(epsilon_input))).clone()
            self.epsilon_output = (epsilon_output.sign() * torch.sqrt(torch.abs(epsilon_output))).clone()
        self.cached_weight = None
        self.cached_bias = None

    def parameter_initialization(self):
        """
        Initialize with normal distribution
        """
        bound = self._calculate_bound()
        self.sigma_bias.data.fill_(value=self.sigma * bound)
        self.sigma_weight.data.fill_(value=self.sigma * bound)
        self.mu_bias.data.uniform_(-bound, bound)
        self.mu_weight.data.uniform_(-bound, bound)

class DDQN(nn.Module):
    """
    Dueling DQN implementation.

    input_dim : state_dim (c, h, w)
    output_dim : output_dim for CNN, should be flattened
    """

    def __init__(self, input_dim, output_dim=256, n_actions=12, lr=0.0001, name='DDQN.ckpt'):
        super(DDQN, self).__init__()
        self.device = "cpu"
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.ckpt_name = name

        self.cnn = self.build_cnn(input_dim[0]) # Just give #of channels
        self.value = NoisyNetLayer(output_dim, 1)
        self.advantage = NoisyNetLayer(output_dim, n_actions)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.to(self.device)

    def forward(self, input):
        flat_val = self.cnn(input)
        value = self.value(flat_val)
        advantage = self.advantage(flat_val)
        return value, advantage

    def build_cnn(self, c):
        cnn = torch.nn.Sequential()
        # Convolution 1
        conv1 = nn.Conv2d(in_channels=c, out_channels=32, kernel_size=4, stride=4)
        nn.init.kaiming_normal_(conv1.weight, mode='fan_out', nonlinearity='relu')
        cnn.add_module("conv_1", conv1)
        cnn.add_module("relu_1", nn.ReLU())

        # Convolution 2
        conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2, stride=2)
        nn.init.kaiming_normal_(conv2.weight, mode='fan_out', nonlinearity='relu')
        cnn.add_module("conv_2", conv2)
        cnn.add_module("relu_2", nn.ReLU())

        # Convolution 3
        conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=2, stride=1)
        nn.init.kaiming_normal_(conv3.weight, mode='fan_out', nonlinearity='relu')
        cnn.add_module("conv_3", conv3)
        cnn.add_module("maxpool", nn.MaxPool2d(kernel_size=2))

        # Reshape CNN output
        class ConvReshape(nn.Module): forward = lambda self, x: x.view(x.size()[0], -1)
        cnn.add_module("reshape", ConvReshape())

        # Calculate input size
        state = torch.zeros(1, *(self.input_dim))
        dims = cnn(state)
        line_input_size = int(np.prod(dims.size()))

        # Linear 1
        line1 = nn.Linear(line_input_size, 512)
        nn.init.kaiming_normal_(conv1.weight, mode='fan_out', nonlinearity='relu')
        cnn.add_module("line_1", line1)
        cnn.add_module("relu_4", nn.ReLU())

        # Linear 2
        line1 = nn.Linear(512, self.output_dim)
        nn.init.kaiming_normal_(conv1.weight, mode='fan_out', nonlinearity='relu')
        cnn.add_module("line_2", line1)
        cnn.add_module("relu_5", nn.ReLU())

        return cnn
    
    def load_ckpt(self, ckpt_dir):
        self.load_state_dict(torch.load(ckpt_dir, map_location=torch.device('cpu')))

class DDDQN(nn.Module):
    """
    Double DDQN implementation
    Target Network will be update with soft approach

    input_dim : state_dim (c, h, w)
    output_dim : output_dim for CNN, should be flattened
    """
    def __init__(self, input_dim, output_dim=256, n_actions=12, ckpt_dir='./ckeckpoints/', name='DDDQN.ckpt'):
        super(DDDQN, self).__init__()
        self.ckpt_dir = ckpt_dir
        self.ckpt_filepath = os.path.join(self.ckpt_dir, name)
        self.device = "cpu"
        self.input_dim = input_dim
        self.output_dim = output_dim

        # self.online_net = DDQN(input_dim=input_dim, output_dim=output_dim, n_actions=n_actions)
        self.target_net = DDQN(input_dim=input_dim, output_dim=output_dim, n_actions=n_actions)
        # self.target_net.load_state_dict(self.online_net.state_dict())
        for p in self.target_net.parameters():
            p.requires_grad = False
        self.to(self.device)
    
    def forward(self, input):
        return self.target_net(input)
    
    def load_ckpt(self, save_dir):
        # save_dir = Path(self.ckpt_dir) / save_dir
        self.target_net.load_ckpt(save_dir)
        # self.online_net.load_ckpt(save_dir)

class Agent():
    def __init__(self):
        self.device = "cpu"
        state_dim = (4, 84, 84)
        self.net = DDDQN(input_dim=state_dim, ckpt_dir='./110062126_hw2_data.py')
        self.net.load_ckpt('./110062126_hw2_data.py')
        self.action_space = [i for i in range(12)]
        # Grayscale, resize, normalize
        self.transforms1 = T.Compose(
            [T.ToTensor(), T.Grayscale()]
        )
        self.transforms2 = T.Compose(
            [T.Resize((84, 84), antialias=True), T.Normalize(0, 255)]
        )
        self.reset()
    
    def reset(self):
        np.random.seed(1116)
        self.timestamp = 0
        self.frame_skip = 0
        self.last_action = None
        self.frames = deque(maxlen=4)

    def act(self, observation):
        """
        Choose action according to Advantage

        Inputs:
            observation (numpy array) : An observation of the current state
        Outputs:
            action_idx (int) : An integer representing the selected action
        """
        # print(self.timestamp)
        if self.timestamp == 3448:
            self.reset()
        if self.frame_skip % 4 == 0:
            observation = self.transforms1(observation.astype('int64').copy())
            observation = self.transforms2(observation.float()).squeeze(0)
            while len(self.frames) < 4:
                self.frames.append(observation)
            self.frames.append(observation)
            observation = gym.wrappers.frame_stack.LazyFrames(list(self.frames))
            observation = observation[0].__array__() if isinstance(observation, tuple) else observation.__array__()
            observation = torch.tensor(observation, device=self.device).unsqueeze(0)
            _, advantage = self.net.target_net.forward(observation)
            prob = nn.Softmax(dim=-1)(advantage/0.3)
            prob = prob.cpu().detach().numpy()[0]
            self.last_action = np.random.choice(self.action_space, p=prob)
        self.frame_skip += 1
        self.timestamp += 1
        
        return self.last_action


if __name__ == '__main__':
    # Create Environment
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, COMPLEX_MOVEMENT)

    agent = Agent()
    tot_reward = 0

    for i in range(50):
        r = 0
        done = False
        state = env.reset()
        start_time = time.time()

        while not done:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)

            if time.time() - start_time > 120:
                break

            tot_reward += reward
            r += reward
            state = next_state
            # env.render('human')
        print(f'Game #{i}: {r}')
    env.close()
    print(f'mean_reward: {tot_reward/50}')