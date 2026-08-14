import os
import torch

import sys
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None
import numpy as np
import pickle
try:
    import cvxpy as cp
except ImportError:
    cp = None
try:
    import jax
    import jax.numpy as jnp
except ImportError:
    jax = None
    jnp = None

def evaluate(states, controls, pos_obs, goal, r):
    """
    Args:
        states: [D, T]
        controls: [D, T]
        pos_obs: [N, D, T]
        goal: [D]
        r: float"""

    distance = torch.norm(states[:2,:].unsqueeze(0)-pos_obs[:,:2,:], dim=1)
    collision = torch.any(distance<r).int()
    
    distance = torch.norm(states[:2,-1] - goal)


    return collision, distance


if jax is not None:
    @jax.jit
    def barrier(r_ab, v_rel, dt):
        dist = jnp.linalg.norm(r_ab)
        
        r_ab_pred = r_ab - v_rel * dt
        pred_dist = jnp.linalg.norm(r_ab_pred)
        v_rel_dt_sq = jnp.dot(v_rel * dt, v_rel * dt)
        
        expr = (dist + pred_dist)**2 - v_rel_dt_sq
        b = 0.5 * jnp.sqrt(jnp.maximum(expr, 1e-6))
        return b

    @jax.jit
    def grad_barrier_exp(r_ab, v_rel, dt):
        def rep_potential(barrier_val, A=2.1, B=0.5):
            return A * jnp.exp(-barrier_val / B)

        V = lambda r: rep_potential(barrier(r, v_rel, dt))
        return jax.grad(V)(r_ab)
else:
    def barrier(r_ab, v_rel, dt):
        r_ab = np.asarray(r_ab, dtype=np.float32)
        v_rel = np.asarray(v_rel, dtype=np.float32)
        dist = np.linalg.norm(r_ab)
        pred_dist = np.linalg.norm(r_ab - v_rel * dt)
        expr = (dist + pred_dist) ** 2 - float(np.dot(v_rel * dt, v_rel * dt))
        return 0.5 * np.sqrt(max(expr, 1e-6))

    def grad_barrier_exp(r_ab, v_rel, dt):
        r_ab = np.asarray(r_ab, dtype=np.float32)
        eps = 1e-3

        def potential(r):
            return 2.1 * np.exp(-barrier(r, v_rel, dt) / 0.5)

        grad = np.zeros_like(r_ab, dtype=np.float32)
        for i in range(r_ab.shape[0]):
            step = np.zeros_like(r_ab, dtype=np.float32)
            step[i] = eps
            grad[i] = (potential(r_ab + step) - potential(r_ab - step)) / (2 * eps)
        return grad

class HumanAgent:
    def __init__(self, robot_goal, radius=0.5, dt=0.1, random_generator=None):
        if random_generator is None:
            random_generator = np.random.RandomState()
        
        self.rng = random_generator
        
        while True:
            self.start = self.rng.uniform(-2, 8, size=(2,))
            if np.linalg.norm(self.start) >= 1.5:
                break
        while True:
            self.goal = self.rng.uniform(-2, 8, size=(2,))
            if np.linalg.norm(self.goal - robot_goal) >= 2.0:
                break
                
        self.radius = radius
        self.sfm_des_speed = self.rng.uniform(0.8, 1.3)
        self.dt = dt
        
        self.state = self.start.copy()
        self.control = np.zeros(2, dtype=np.float32)

    def social_force_step(self, others_states, others_controls, tau=0.5):
        if np.linalg.norm(self.goal - self.state) < 0.1:
            self.control = np.zeros(2, dtype=np.float32)
            return

        goal_direction = (self.goal - self.state) / np.linalg.norm(self.goal - self.state)
        desired_velocity = self.sfm_des_speed * goal_direction
        
        goal_force = (1/tau) * (desired_velocity - self.control)

        repulsive_force = np.zeros(2, dtype=np.float32)
        for o_x, o_u in zip(others_states, others_controls):

            r_ab = self.state - o_x
            v_rel = self.control - o_u
            
            if jnp is not None:
                grad = grad_barrier_exp(jnp.array(r_ab), jnp.array(v_rel), self.dt)
            else:
                grad = grad_barrier_exp(np.array(r_ab), np.array(v_rel), self.dt)
            repulsive_force += -1.0 * np.array(grad)

        total_acceleration = goal_force + repulsive_force
        
        self.control += total_acceleration * self.dt
        
        speed = np.linalg.norm(self.control)
        if speed > self.sfm_des_speed * 1.3:
             self.control = (self.control / speed) * (self.sfm_des_speed * 1.3)

        self.state += self.control * self.dt



class AgentHistory:
    def __init__(self, max_length: int):
        self.max_length = max_length
        self.data = None
        
    def update(self, new_data: torch.Tensor):
        new_data = new_data.unsqueeze(-1)  # [..., D] -> [..., D, 1]
        if self.data is None:
            self.data = new_data
        else:
            if len(self) < self.max_length:
                self.data = torch.cat([self.data, new_data], dim=-1)
            else:
                self.data = torch.cat([self.data[..., 1:], new_data], dim=-1)
                
    def get(self):
        return self.data
        
    def __len__(self):
        return self.data.shape[-1] if self.data is not None else 0
