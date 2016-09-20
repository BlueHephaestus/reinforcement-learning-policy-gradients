import numpy as np

discount_factor = 0.95

#baseline2 = 0
rewards = []
baselines = []

for t in range(100):
  reward = 1*discount_factor**(t)
  rewards.append(reward)
  baseline = np.mean(rewards)
  baselines.append(baseline)
  #baseline2 += reward/100.0

  #print reward, baseline1, baseline2, reward-baseline1, reward-baseline2
  print reward-baseline



