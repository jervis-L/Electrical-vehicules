from env import EVs
from brain import DeepQNetwork
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#tf.compat.v1.disable_eager_execution()
action_space = {0:'000',1:'001',2:'002',3:'010',4:'011',5:'012',6:'020',7:'021',8:'022',\
            9:'100',10:'101',11:'102',12:'110',13:'111',14:'112',15:'120',16:'121',\
                17:'200',18:'201',19:'202',20:'210',21:'211',22:'220'}
def run_EVs():
    step = 0
    for episode in range(100):
        # initial observation
        observation = env.reset()

        while True:
            
            # RL choose action based on observation
            action = RL.choose_action(observation)
            if episode ==0:
                print(action_space[action])
            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)

            RL.store_transition(observation, action, reward, observation_)

            if (step > 200) and (step % 5 == 0):
                RL.learn()

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                break
            step += 1

    # end of game
    print('game over')



if __name__ == "__main__":
    # maze game
    env = EVs()
    RL = DeepQNetwork(env.n_actions, env.n_features,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=2000,
                      # output_graph=True
                      )
    run_EVs()
    # env.after(100, run_EVs)
    # env.mainloop()
    # print(env.actions[100:110])

