import numpy as np 
import torch 
import gym
from agent import Agent

def TrainModel(agent):
    from collections import deque
    number_episozed = 2000
    maximum_number_timesteps_per_episodes = 1000
    epssilon_starting_value = 1.0 
    epssilon_ending_value = 0.01
    epssilon_decay_value = 0.995
    epsilon = epssilon_starting_value
    scores_on_100_episodes = deque(maxlen=100)

    for episode in range(1, number_episozed+1):
        state, _ = env.reset()
        score = 0 
        for _ in range(maximum_number_timesteps_per_episodes):
            action = agent.act(state, epsilon)
            next_state, reward, done, _ , _ = env.step(action)
            agent.step(state,action,reward,next_state,done)
            state = next_state
            score += reward
            if done: 
                break
        
        scores_on_100_episodes.append(score)
        epsilon = max(epssilon_ending_value, epssilon_decay_value*epsilon)
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode,np.mean(scores_on_100_episodes)), end="")
        if episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode,np.mean(scores_on_100_episodes)))
        if np.mean(scores_on_100_episodes) >= 200.0:
            print('\Envierment solved in  {:d} episoed!\tAverage Score: {:.2f}'.format(episode-100,np.mean(scores_on_100_episodes)))
            torch.save(agent.local_qnetork.state_dict(), 'checkpoint.pth')
            break


def LoadModel(agent, model_file_path = 'checkpoint.pth'):
    import os
    agent.local_qnetork.load_state_dict(torch.load(model_file_path))
        

def ShowResult(agent):
    import glob
    import io
    import base64
    import imageio
    from IPython.display import HTML, display

    def show_video_of_model(agent, env_name):
        env = gym.make(env_name, render_mode='rgb_array')
        state, _ = env.reset()
        done = False
        frames = []
        while not done:
            frame = env.render()
            frames.append(frame)
            action = agent.act(state)
            state, reward, done, _, _ = env.step(action.item())
        env.close()
        imageio.mimsave('video.mp4', frames, fps=30)

    show_video_of_model(agent, 'LunarLander-v2')

    def show_video():
        mp4list = glob.glob('*.mp4')
        if len(mp4list) > 0:
            mp4 = mp4list[0]
            video = io.open(mp4, 'r+b').read()
            encoded = base64.b64encode(video)
            display(HTML(data='''<video alt="test" autoplay
                    loop controls style="height: 400px;">
                    <source src="data:video/mp4;base64,{0}" type="video/mp4" />
                </video>'''.format(encoded.decode('ascii'))))
        else:
            print("Could not find video")

    show_video()


if __name__ == "__main__":
    env = gym.make('LunarLander-v2')
    state_shape = env.observation_space.shape
    state_size =  env.observation_space.shape[0]
    number_actions = env.action_space.n
    agent = Agent(state_size, number_actions)

    TrainModel(agent)
    # LoadModel(agent)
    ShowResult(agent)