from stable_baselines.common.callbacks import BaseCallback
import numpy as np

class CustomCallback(BaseCallback):
    
    def __init__(self):
        super(CustomCallback, self).__init__()
        self.count = 0
    
    def _on_step(self):

        print(self.locals)
        
        if self.training_env.done:
            self.count += 1
            reward = self.locals['episode_rewards'][-1]
            print('Episode: ' + str(self.count) + ' | Reward: ' + str(reward))

        
        return True

    def _on_rollout_end(self):
        """
        This event is triggered before updating the policy.
        """
        pass

    def _on_training_end(self):
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass
    
    def _on_training_start(self):
        """
        This method is called before the first rollout starts.
        """
        pass

    def _on_rollout_start(self):
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

class EvaluateCallbackInstance:

    def __init__(self):
        self.rewards_arrays = []
        self.reward_array_run = []
        self.result = None
        self.ccy_dims = 0
        self.local = None
        self.count = 0

    def evaluateCallback(self, locals_, globals_):

        if self.count == 0 :
            self.local = np.copy(locals_["obs"])
            self.count += 1

        self.reward_array_run.append(locals_["reward"][0])
        
        if locals_["done"]:
            self.rewards_arrays.append(self.reward_array_run)
            self.reward_array_run = []