import gym
env = gym.make('CartPole-v0')

env.reset()

species_num = 10

generations = []

for generation_i in range(1000):
    observation = env.reset()

    #Use same method that chooses random action if not provided in the generation to make our entire first generation.
    #Idk how yet but gtg anodium
    for species_i, species in enumerate(generations):
        t = 0
        while True:#Run the species
            env.render()#Update our visual environment
            #print(observation)

            #Randomly get next action
            action = env.action_space.sample()

            #get the result of our action for next timestep
            observation, reward, done, info = env.step(action)

            if done:
                #Exit this species and go next
                print("Species finished after {} timesteps".format(t+1))#End
                break
            #Increment timestep
            t+=1
