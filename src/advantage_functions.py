import numpy as np

class updated_mean(object):

    @staticmethod
    def get_advantage(rewards, reward):
        #Get baseline as continuously updated average of rewards
        baseline = np.mean(rewards)

        #Discounted reward - baseline function output for t
        advantage = reward - baseline

        return advantage

