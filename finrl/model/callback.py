from stable_baselines.common.callbacks import BaseCallback

class Callback(BaseCallback):
    
    def __init__(self, model):
        super(Callback, self).__init__()
        self.count = 0
        self.model = model
    
    def _on_step(self) -> bool:
        
        if self.training_env.done:
            self.count += 1
            reward = self.locals['episode_rewards'][-1]
            print('Episode: ' + str(self.count) + ' | Reward: ' + str(reward))

        
        return True