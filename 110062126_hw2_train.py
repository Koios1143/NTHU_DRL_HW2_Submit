
import gym.spaces
import torch
from torch import nn
from collections import deque
from torchvision import transforms as T

import gym
from gym.spaces import Box
import gym_super_mario_bros
from gym.wrappers import FrameStack
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT

import numpy as np
from pathlib import Path
import os, time, datetime
from tqdm import trange
import wandb

# Add wrappers to do the followings
# 1. SkipFrame: perform same action for n-frames
# 2. GrayScale: To ignore lots color info, gray scale is enough
# 3. DownSample: We don't need too detail info, maybe (256, 240) -> (128, 128) is enough
# 4. FrameStack: Stacks the observations

class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip=4):
        super().__init__(env)
        self._skip = skip
    
    def step(self, action):
        total_reward = 0.0
        for _ in range(self._skip):
            obs, reward, end, info = self.env.step(action)
            total_reward += reward
            if end:
                break
        return obs, total_reward, end, info

class GrapScale(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = Box(low=0, high=255, shape=self.observation_space.shape[:2], dtype=np.uint8)
    
    def permute_orientation(self, obs):
        # Since torchvision use [C, H, W] rather than [H, W, C], we should transform it first
        return T.ToTensor()(obs.astype('int64').copy())

    def observation(self, obs):
        obs = self.permute_orientation(obs)
        obs = T.Grayscale()(obs)
        return obs

class DownSample(gym.ObservationWrapper):
    def __init__(self, env, shape=(84, 84)):
        super().__init__(env)
        self._shape = shape + self.observation_space.shape[2:] # Add the third axis
        self.observation_space = Box(low=0, high=255, shape=self._shape, dtype=np.uint8)
    
    def observation(self, obs):
        # Normalize because there would be FrameStack later on
        transforms = T.Compose(
            [T.Resize(self._shape, antialias=True), T.Normalize(0, 255)]
        )
        obs = transforms(obs.float()).squeeze(0)
        return obs

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
        """
        Forward pass the layer. If training, sample noise depends on sample_noise.
        Otherwise, use the default weight and bias.
        """
        if self.training:
            if sample_noise:
                self.sample_noise()
            return nn.functional.linear(x, weight=self.weight, bias=self.bias)
        else:
            return nn.functional.linear(x, weight=self.mu_weight, bias=self.mu_bias)

    def register_noise_buffers(self):
        """
        Register noise f(epsilon_in) and f(epsilon_out)
        """
        self.register_buffer(name='epsilon_input', tensor=torch.empty(self.in_features))
        self.register_buffer(name='epsilon_output', tensor=torch.empty(self.out_features))

    def _calculate_bound(self):
        """
        Determines the initialization bound for the FactorisedNoisyLayer based on the inverse
        square root of the number of input features. This approach to determining the bound
        takes advantage of the factorised noise model's efficiency and aims to balance the
        variance of the outputs relative to the variance of the inputs. Ensuring that the
        initialization of weights does not saturate the neurons and allows for stable
        gradients during the initial phases of training.
        """
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
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.ckpt_name = name

        self.cnn = self.build_cnn(input_dim[0]) # Just give #of channels

        """
        Here we choose NoisyNet, if not, then just linear
        """
        # self.value = nn.Linear(output_dim, 1)
        # self.advantage = nn.Linear(output_dim, n_actions)
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
        line2 = nn.Linear(512, self.output_dim)
        nn.init.kaiming_normal_(conv1.weight, mode='fan_out', nonlinearity='relu')
        cnn.add_module("line_2", line2)
        cnn.add_module("relu_5", nn.ReLU())

        return cnn
    
    def save_ckpt(self, ckpt_dir, code):
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        self.ckpt_filepath = os.path.join(ckpt_dir, self.ckpt_name.format(code))
        torch.save(self.state_dict(), self.ckpt_filepath)
        print('checkpoint saved')
    
    def load_ckpt(self, ckpt_dir, code):
        ckpt_filepath = os.path.join(str(ckpt_dir), self.ckpt_name.format(code))
        self.load_state_dict(torch.load(ckpt_filepath))
        print('ckeckpoint loaded')

class DDDQN(nn.Module):
    """
    Double DDQN implementation

    input_dim : state_dim (c, h, w)
    output_dim : output_dim for CNN, should be flattened
    """
    def __init__(self, lr, input_dim, output_dim=256, n_actions=12, ckpt_dir='./ckeckpoints/', name='DDDQN.ckpt'):
        super(DDDQN, self).__init__()
        self.ckpt_dir = ckpt_dir
        self.ckpt_filepath = os.path.join(self.ckpt_dir, name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Double Networks
        self.online_net = DDQN(input_dim=input_dim, output_dim=output_dim, n_actions=n_actions, name='DDQN_online_{}.ckpt', lr=lr)
        self.target_net = DDQN(input_dim=input_dim, output_dim=output_dim, n_actions=n_actions, name='DDQN_target_{}.ckpt', lr=lr)
        self.target_net.load_state_dict(self.online_net.state_dict())
        for p in self.target_net.parameters():
            p.requires_grad = False
        self.to(self.device)
    
    def forward(self, input, model):
        if model == 'online':
            return self.online_net(input)
        else:
            return self.target_net(input)
    
    def save_ckpt(self, code):
        self.target_net.save_ckpt(ckpt_dir=self.ckpt_dir, code=code)
        self.online_net.save_ckpt(ckpt_dir=self.ckpt_dir, code=code)
    
    def load_ckpt(self, save_dir, code):
        # save_dir = Path(self.ckpt_dir) / save_dir
        self.target_net.load_ckpt(save_dir, code)
        self.online_net.load_ckpt(save_dir, code)

class ReplayBuffer():
    """
    Prioritize Experience Replay (PER) implementation
    """
    def __init__(self, max_size, input_dim, n_actions, save_dir):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.priorities = deque(maxlen=max_size)
        self.save_dir = save_dir

        self.state_memory = np.zeros((self.mem_size,*input_dim), dtype=np.float32)
        self.new_state_memory =np.zeros((self.mem_size,*input_dim), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size,dtype=np.int64)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool_)
    
    def store(self, state, action, reward, next_state, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = next_state
        self.terminal_memory[index] = done
        self.priorities.append(max(self.priorities, default=1))
        self.mem_cntr += 1
    
    def get_probabilities(self):
        """
        P(i) = (p_i^alpha) / sum_i(p_i^alpha)
        """
        scaled_priorities = np.array(self.priorities)
        scaled_priorities = scaled_priorities/ scaled_priorities.sum()
        return scaled_priorities

    def get_importance(self, probabilities, beta):
        """
        Importance-Sampling
            W = ((1/N) * (1/P)) ** beta

        [NOTE] For stability reasons, we always normalize weights by 1/max(importance) so that they only scale the update downwards.
        """
        self.beta = beta
        importance =  np.power(1/self.mem_size * 1/probabilities, -self.beta)
        importance = importance / max(importance)
        return importance

    def sample(self, batch_size, beta):
        max_mem = min(self.mem_cntr, self.mem_size)
        sample_probs = self.get_probabilities()
        batch = np.random.choice(max_mem, batch_size, replace=False, p= sample_probs)
        
        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        next_states = self.new_state_memory[batch]
        dones = self.terminal_memory[batch]
        importance = self.get_importance(sample_probs[batch], beta)

        return states, actions, rewards, next_states, dones, importance, batch
    
    def set_priorities(self, idx, delta, epsilon=1.1, alpha = 0.7):
        """
        p_i^alpha = (|delta_i| + epsilon) ** alpha
        """
        self.priorities[idx] = (np.abs(delta) + epsilon)** alpha

class Agent():
    def __init__(self, state_dim, action_num=12, replace_step=10000, beta=0.4, gamma=0.9,
                 temperature=0.2, temp_min=0.0004, temp_dec = 1e-6, lr=0.0001, save_every=5e5, batch_size=128,
                 save_dir='./ckeckpoints/', load_dir='', last_code=0):
        self.state_dim = state_dim
        self.action_num = action_num
        self.save_dir = save_dir
        self.load_dir = load_dir
        self.last_code = last_code
        self.action_space = [i for i in range(self.action_num)]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Hyperparameters
        self.total_step_count = 0
        self.learn_step_count = 0
        self.replace_step = replace_step
        self.save_every = save_every
        self.gamma = gamma              # For Double DQN
        self.beta = beta                # For PER
        self.temperature = temperature  # For action softmax
        self.temp_min = temp_min        # For action softmax
        self.temp_dec = temp_dec        # For action softmax
        self.lr = lr                    # For optimizer

        self.net = DDDQN(lr=self.lr, input_dim=state_dim, ckpt_dir=self.save_dir)

        # For cache
        self.memory = ReplayBuffer(max_size=100000, input_dim=self.state_dim, n_actions=self.action_num, save_dir=self.save_dir)
        self.batch_size = batch_size

    def select_action(self, state):
        """
        Sample action according to Advantage
        """
        state = state[0].__array__() if isinstance(state, tuple) else state.__array__()
        state = torch.tensor(state, device=self.device).unsqueeze(0)
        _, advantage = self.net.online_net.forward(state)
        prob = nn.Softmax(dim=-1)(advantage/self.temperature)
        prob = prob.cpu().detach().numpy()[0]
        action = np.random.choice(self.action_space, p=prob)

        self.total_step_count += 1
        return action

    def store(self, state, action, reward, next_state, done):
        """
        Save experience to replay buffer
        """
        self.memory.store(state, action, reward, next_state, done)
        
    def sample(self):
        """
        Sample experience from replay buffer
        """
        self.beta = np.min([1., self.beta+0.001])
        state, action, reward, next_state, done, importance, batch = self.memory.sample(self.batch_size, self.beta)  

        states = torch.tensor(state).to(self.device)
        actions = torch.tensor(action).to(self.device)
        rewards = torch.tensor(reward).to(self.device)
        next_states = torch.tensor(next_state).to(self.device)
        dones = torch.tensor(done).to(self.device)
        importance = torch.tensor(importance, dtype=torch.float32).to(self.device)
        batch = torch.tensor(batch).to(self.device)
        return states, actions, rewards, next_states, dones, importance, batch
    
    def sync_target(self):
        """
        Update target network. Hard update.
        """
        if self.learn_step_count % self.replace_step ==0:
            self.net.target_net.load_state_dict(self.net.online_net.state_dict())

    def decrement_temperature(self):
        self.temperature = self.temperature - self.temp_dec if self.temperature > self.temp_min else self.temp_min

    def save(self):
        code = self.learn_step_count // self.save_every
        self.net.save_ckpt(code)
        # [Note] We don't save the replay buffer. For finetuning, just fill the buffer before finetune.
    
    def load(self):
        self.net.load_ckpt(self.load_dir, self.last_code)
    
    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return (-1, -1)

        self.net.online_net.optimizer.zero_grad()
        self.sync_target()

        states, actions, rewards, next_states, dones, importance, batch = self.sample()  
        indices = np.arange(self.batch_size)

        # Value shape (batch,)
        # Advantage shape (batch,action_space.n)
        value_s, adv_s = self.net.online_net.forward(states)
        value_next_s, adv_next_s = self.net.target_net.forward(next_states)
        value_next_s_eval, adv_next_s_eval = self.net.online_net(next_states)

        """
        TD Estimate (in dueling)
            TD_e = Q_online^*(s, a) = V(s) + (A(s, a) - 1/|A| * \sum_a' A(s, a'))
        """
        q_pred = torch.add(value_s, (adv_s - adv_s.mean(dim=1, keepdim=True)))[indices, actions]
        q_next = torch.add(value_next_s,(adv_next_s - adv_next_s.mean(dim=1, keepdim=True)))
        q_eval = torch.add(value_next_s_eval, (adv_next_s_eval-adv_next_s_eval.mean(dim=1, keepdim=True)))
        max_actions = torch.argmax(q_eval, dim=1)

        # masking terminal states
        q_next[dones] = 0.0
        # Bellman equation
        """
        TD target
            a' = argmax_a Q_online(s', a)
            TD_t = r + gamma * Q_target^*(s', a')
        """
        q_target = rewards + self.gamma * q_next[indices, max_actions]

        # Temporal-difference Error term for prioritized experience replay
        diff = torch.abs(q_pred - q_target)
        for i in range(self.batch_size):
            idx = batch[i]
            self.memory.set_priorities(idx, diff[i].cpu().detach().numpy())

        loss = (torch.cuda.FloatTensor(importance) * torch.nn.functional.smooth_l1_loss(q_pred, q_target)).mean().to(self.device)
        loss.backward()

        self.net.online_net.optimizer.step()
        self.learn_step_count +=1
        self.decrement_temperature()

        # Save model
        if self.learn_step_count % self.save_every == 0:
            self.save()
        
        return (q_pred.mean().item(), loss.item())
    
class MetricLogger:
    def __init__(self, save_dir, run=None):
        self.save_log = save_dir / "log"
        with open(self.save_log, "w") as f:
            f.write(
                f"{'Episode':>8}{'Step':>8}{'MeanReward':>15}"
                f"{'MeanLength':>15}{'MeanLoss':>15}{'MeanQValue':>15}"
                f"{'TimeDelta':>15}{'Time':>20}\n"
            )

        # History metrics
        self.ep_rewards = []
        self.ep_lengths = []
        self.ep_avg_losses = []
        self.ep_avg_qs = []

        # Moving averages, added for every call to record()
        self.moving_avg_ep_rewards = []
        self.moving_avg_ep_lengths = []
        self.moving_avg_ep_avg_losses = []
        self.moving_avg_ep_avg_qs = []

        # Current episode metric
        self.init_episode()

        # Timing
        self.record_time = time.time()

        # Wandb
        self.run = run

    def init_episode(self):
        self.curr_ep_reward = 0.0
        self.curr_ep_length = 0
        self.curr_ep_loss = 0.0
        self.curr_ep_q = 0.0
        self.curr_ep_loss_length = 0

    def log_step(self, reward, loss, q):
        self.curr_ep_reward += reward
        self.curr_ep_length += 1
        self.curr_ep_loss += loss
        self.curr_ep_q += q
        self.curr_ep_loss_length += 1

    def log_episode(self):
        "Mark end of episode"
        self.ep_rewards.append(self.curr_ep_reward)
        self.ep_lengths.append(self.curr_ep_length)
        if self.curr_ep_loss_length == 0:
            ep_avg_loss = 0
            ep_avg_q = 0
        else:
            ep_avg_loss = np.round(self.curr_ep_loss / self.curr_ep_loss_length, 5)
            ep_avg_q = np.round(self.curr_ep_q / self.curr_ep_loss_length, 5)
        self.ep_avg_losses.append(ep_avg_loss)
        self.ep_avg_qs.append(ep_avg_q)

        self.init_episode()

    def record(self, episode, step):
        mean_ep_reward = np.round(np.mean(self.ep_rewards[-100:]), 3)
        mean_ep_length = np.round(np.mean(self.ep_lengths[-100:]), 3)
        mean_ep_loss = np.round(np.mean(self.ep_avg_losses[-100:]), 3)
        mean_ep_q = np.round(np.mean(self.ep_avg_qs[-100:]), 3)
        self.moving_avg_ep_rewards.append(mean_ep_reward)
        self.moving_avg_ep_lengths.append(mean_ep_length)
        self.moving_avg_ep_avg_losses.append(mean_ep_loss)
        self.moving_avg_ep_avg_qs.append(mean_ep_q)

        last_record_time = self.record_time
        self.record_time = time.time()
        time_since_last_record = np.round(self.record_time - last_record_time, 3)

        print(
            f"Episode {episode} - "
            f"Step {step} - "
            f"Mean Reward {mean_ep_reward} - "
            f"Mean Length {mean_ep_length} - "
            f"Mean Loss {mean_ep_loss} - "
            f"Mean Q Value {mean_ep_q} - "
            f"Time Delta {time_since_last_record} - "
            f"Time {datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}"
        )

        if self.run != None:
            self.run.log({
                "Episode": episode,
                "Step": step,
                "Mean Reward": mean_ep_reward,
                "Mean Length": mean_ep_length,
                "Mean Loss": mean_ep_loss,
                "Mean Q Value": mean_ep_q,
                "Time Delta": time_since_last_record
            })

        with open(self.save_log, "a") as f:
            f.write(
                f"{episode:8d}{step:8d}"
                f"{mean_ep_reward:15.3f}{mean_ep_length:15.3f}{mean_ep_loss:15.3f}{mean_ep_q:15.3f}"
                f"{time_since_last_record:15.3f}"
                f"{datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'):>20}\n"
            )

if __name__ == '__main__':
    # Create Environment
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    env = SkipFrame(env)
    env = GrapScale(env)
    env = DownSample(env)
    env = FrameStack(env, num_stack=4)

    # Record arguments
    best_score = -np.inf
    prev_avg = -np.inf
    load_checkpoint = False
    save_dir = Path("checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    save_dir.mkdir(parents=True)
    load_dir = None #'./checkpoints/2024-04-01T20-04-08/'
    load_code = 0 #8

    # Hyperparameters
    n_episodes = 1000000
    replace_step=10000
    beta=0.4
    gamma=0.9
    temperature=0.2
    temp_min=0.0004
    temp_dec = 1e-7
    lr=0.0001
    save_every=10000
    batch_size=64

    agent = Agent(state_dim=(env.observation_space.shape), action_num=env.action_space.n, 
                  replace_step=replace_step, beta=beta, gamma=gamma, temperature=temperature, temp_min=temp_min,
                  temp_dec=temp_dec, lr=lr, save_every=save_every, batch_size=batch_size,
                  save_dir=save_dir, load_dir=load_dir, last_code=load_code)
    
    run = wandb.init(
        project="DRL-HW2",
        # Track hyperparameters and run metadata
        config={
            "replace_step": replace_step,
            "beta": beta,
            "gamma": gamma,
            "temperature": temperature,
            "temp_min": temp_min,
            "temp_dec": temp_dec,
            "learning_rate": lr,
            "save_every": save_every,
            "batch_size": batch_size
        },
    )

    logger = MetricLogger(save_dir, run)

    if load_checkpoint:
        agent.load()

    # Training loop
    for episode in trange(n_episodes):
        done = False
        state = env.reset()

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)

            agent.store(state, action, reward, next_state, done)
            q, loss = agent.learn()
            logger.log_step(reward, loss, q)

            state = next_state
        
        logger.log_episode()
        if episode % 20 == 0:
            logger.record(episode, agent.total_step_count)
