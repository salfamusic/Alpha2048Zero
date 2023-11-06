import numpy as np

from PPOAgent import PPOAgent
import tensorflow as tf
import keras.backend as K

class VectorizedPPOAgent(PPOAgent):
    def vectorized_predict(self, states, legal_moves_masks):
        logits, values = self.model(self.log2(states.reshape(-1, 4, 4, 5)))
        logits_array = logits.numpy()
        logits_array /= self.temperature

        # Apply mask to logits by setting illegal move logits to a large negative value
        for i, legal_moves_mask in enumerate(legal_moves_masks):
            logits_array[i][~legal_moves_mask] = -1e10

        probs = tf.nn.softmax(logits_array).numpy()

        return values[:, 0], probs

    def vectorized_select_action(self, states, legal_moves_masks):
        _, probs_batch = self.vectorized_predict(states, legal_moves_masks)
        actions = [np.random.choice(self.action_space, p=probs) for probs in probs_batch]
        return actions

    def vectorized_train(self, states, actions, rewards, next_states, old_logits, dones):
        # Increment the step counter
        self.step += 1


        # Update the entropy coefficient and learning rate
        self.entropy_coeff = self._decay_entropy_coeff()
        self.lr = self._decay_lr()
        K.set_value(self.model.optimizer.lr, self.lr)
        
        states = self.log2(np.array(states).reshape((-1, 4, 4, 5)))
        actions = np.array(actions).reshape((-1, 1))
        rewards = np.array(rewards).reshape((-1, 1))
        next_states = self.log2(np.array(next_states).reshape((-1, 4, 4, 5)))
        old_logits = np.array(old_logits).reshape((-1, 4))
        dones = np.array(dones).reshape((-1, 1))

        values = self.model.predict(states, verbose=0)[1]
        next_values = self.model.predict(next_states, verbose=0)[1]

        targets = rewards + self.gamma * next_values * (1. - dones)
        advantages = targets - values

        y_true_policy = np.concatenate([old_logits, advantages, actions], axis=-1)

        loss_metrics = self.model.train_on_batch(states, [y_true_policy, targets])
        ppo_loss, value_loss, _ = loss_metrics
        print(f'PPO Loss: {ppo_loss:.4f}, Value Loss: {value_loss:.4f}')