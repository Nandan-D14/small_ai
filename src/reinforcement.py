import torch
import torch.nn as nn
import torch.optim as optim
import random
import json

class ReinforcementAgent:
    def __init__(self, model, vocab_size, lr=0.001, gamma=0.99):
        self.model = model
        self.vocab_size = vocab_size
        self.gamma = gamma  # Discount factor
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()

    def select_action(self, state):
        """Select an action using the model (policy)."""
        with torch.no_grad():
            output = self.model(state)
        action = torch.argmax(output, dim=-1)
        return action

    def train(self, experiences):
        """Train the model using reinforcement learning experiences."""
        for state, action, reward, next_state in experiences:
            state = torch.tensor(state, dtype=torch.long)
            action = torch.tensor(action, dtype=torch.long)
            reward = torch.tensor(reward, dtype=torch.float32)
            next_state = torch.tensor(next_state, dtype=torch.long)

            self.optimizer.zero_grad()
            
            # Predict current state action value
            output = self.model(state)
            predicted_q_value = output.gather(1, action.unsqueeze(1)).squeeze(1)

            # Predict next state max Q-value
            with torch.no_grad():
                next_output = self.model(next_state)
                max_next_q_value = torch.max(next_output, dim=-1)[0]

            # Compute target Q-value
            target_q_value = reward + self.gamma * max_next_q_value
            loss = self.criterion(predicted_q_value, target_q_value)

            loss.backward()
            self.optimizer.step()

    def save_experiences(self, experiences, file_path):
        """Save experiences to a JSON file."""
        with open(file_path, 'w') as f:
            json.dump(experiences, f)

    def load_experiences(self, file_path):
        """Load experiences from a JSON file."""
        with open(file_path, 'r') as f:
            experiences = json.load(f)
        return experiences
