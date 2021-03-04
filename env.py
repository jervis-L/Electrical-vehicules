import numpy as np
import time
import sys
if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk
# initial observation observation = env.reset()
# fresh env env.render()
# RL take action and get next observation and reward observation_, reward, done = env.step(action)


class EVs(tk.Tk,object):

    def __init__(self):
        super(EVs, self).__init__()
        self.nev=3
        self.n_time_int=10
        self.timeslotval=1
        self.max_rate=4
        self.ener=5
        self.EV_state=np.zeros(2*self.nev+1)
        self.actions=[]
        # 23 actions(3x3x3-4)
        self.action_space = {0:'000',1:'001',2:'002',3:'010',4:'011',5:'012',6:'020',7:'021',8:'022',\
            9:'100',10:'101',11:'102',12:'110',13:'111',14:'112',15:'120',16:'121',\
                17:'200',18:'201',19:'202',20:'210',21:'211',22:'220'}
        self.cost_1=0.1558
        self.cost_2=0.1599
        self.p=np.zeros(self.nev)
        # params used in DQN network
        self.n_actions = len(self.action_space)
        self.n_features = 7
    def createEV(self):
        for i in range(self.nev):
            self.EV_state[i]=5
            self.EV_state[i+self.nev]=10

    def cal_power(self,EV_state,EV_state_):
        for i in range(self.nev):
            self.p[i]=EV_state[i]-EV_state_[i]

    def profit(self):
        cost=sum(self.p)*self.cost_1
        return -cost

    def smoothing(self):
        pass
        

    def equal_sharing(self):
        return -(max(self.p)-min(self.p))**2

    def quickcharging(self):
        timeleft=self.EV_state[self.nev:2*self.nev]
        return np.dot(timeleft,self.p)

    def utility(self):
        # utility function needs to be changed
        
        return self.profit()+self.equal_sharing()+self.quickcharging()

    def reset(self):
        # set utility under initial strategy
        self.createEV()
        
        return np.array(self.EV_state)
    def step(self,action):
        action=self.action_space[action]
        self.actions.append(action)
        base_action=[0,0,0]
        reward=0
        s=self.EV_state
        for i in range(len(action)):
            base_action[i]=int(action[i])
        for i in range(self.nev):
            self.EV_state[i]-=base_action[i]*self.timeslotval
            self.EV_state[i+self.nev]-=1
            self.EV_state[-1]+=1
        s_=self.EV_state
        self.cal_power(s,s_)
        u=self.utility()
        # reward function
        # total power < max provided power
        # charging rate < max_rate
        if u>=2:
            reward= 1
        if 2<u<1:
            reward= 0
        if u<=1:
            reward=-1
        if self.EV_state[-1]==self.n_time_int:
            done=True
        else:
            done=False
        return self.EV_state, reward,done
    def render(self):
        self.update()

