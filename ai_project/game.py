from keras.models import model_from_json
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD , Adam
from keras.callbacks import TensorBoard
from collections import deque
import numpy as np
import time
import random
from utils import Utils
import os
import pandas as pd
import json
from IPython.display import clear_output
from action import Action
from agent import Agent

class Game:

    chrome_driver_path = "../chromedriver"
    loss_file_path = "./objects/loss_df.csv"
    actions_file_path = "./objects/actions_df.csv"
    q_value_file_path = "./objects/q_values.csv"
    scores_file_path = "./objects/scores_df.csv"
    loss_df = pd.read_csv(loss_file_path) if os.path.isfile(loss_file_path) else pd.DataFrame(columns =['loss'])
    scores_df = pd.read_csv(scores_file_path) if os.path.isfile(loss_file_path) else pd.DataFrame(columns = ['scores'])
    actions_df = pd.read_csv(actions_file_path) if os.path.isfile(actions_file_path) else pd.DataFrame(columns = ['actions'])
    q_values_df =pd.read_csv(actions_file_path) if os.path.isfile(q_value_file_path) else pd.DataFrame(columns = ['qvalues'])
    #game parameters
    ACTIONS = 2 # possible actions: jump, do nothing
    GAMMA = 0.99 # decay rate of past observations original 0.99
    OBSERVATION = 100. # timesteps to observe before training
    EXPLORE = 100000  # frames over which to anneal epsilon
    FINAL_EPSILON = 0.1 # final value of epsilon
    INITIAL_EPSILON = 0.1 # starting value of epsilon
    REPLAY_MEMORY = 50000 # number of previous transitions to remember
    BATCH = 16 # size of minibatch
    FRAME_PER_ACTION = 1
    LEARNING_RATE = 1e-4
    img_rows , img_cols = 80,80
    img_channels = 4 #We stack 4 frames 
    game_url = "chrome://dino"  
    
    def __init__(self, agent,game):
        self._agent = agent
        self._game = game
        self._utils = Utils()
        self._display = self._utils.show_img() #display the processed image on screen using openCV, implemented using python coroutine 
        self._display.__next__() # initiliaze the display coroutine
    # training variables saved as checkpoints to filesystem to resume training from the same step
    def init_cache(self):
        """initial variable caching, done only once"""
        self._utils.save_obj(self.INITIAL_EPSILON,"epsilon")
        t = 0
        self._utils.save_obj(t,"time")
        D = deque()
        self._utils.save_obj(D,"D")

    def get_state(self,actions):
        self.actions_df.loc[len( self.actions_df)] = actions[1] # storing actions in a dataframe
        score = self._game.get_score() 
        reward = 0.1
        is_over = False #game over
        if actions[1] == 1:
            self._agent.jump()
        image = self._utils.grab_screen(self._game._driver) 
        self._display.send(image) #display the image on screen
        if self._agent.is_crashed():
            self.scores_df.loc[len(self.loss_df)] = score # log the score when game is over
            self._game.restart()
            reward = -1
            is_over = True
        return image, reward, is_over #return the Experience tuple

    def buildmodel(self):
        print("Now we build the model")
        model = Sequential()
        model.add(Conv2D(32, (8, 8), padding='same',strides=(4, 4),input_shape=(self.img_cols,self.img_rows,self.img_channels)))  #80*80*4
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (4, 4),strides=(2, 2),  padding='same'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (3, 3),strides=(1, 1),  padding='same'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dense(self.ACTIONS))
        adam = Adam(lr=self.LEARNING_RATE)
        model.compile(loss='mse',optimizer=adam)
        
        #create model file if not present
        if not os.path.isfile(self.loss_file_path):
            model.save_weights('model.h5')
        print("We finish building the model")
        return model

    def trainNetwork(self,model,game_state,observe=False):
        last_time = time.time()
        # store the previous observations in replay memory
        D = self._utils.load_obj("D") #load from file system
        # get the first state by doing nothing
        do_nothing = np.zeros(self.ACTIONS)
        do_nothing[0] =1 #0 => do nothing,
                        #1=> jump
        
        x_t, r_0, terminal = game_state.get_state(do_nothing) # get next step after performing the action
        

        s_t = np.stack((x_t, x_t, x_t, x_t), axis=2) # stack 4 images to create placeholder input
        

        
        s_t = s_t.reshape(1, s_t.shape[0], s_t.shape[1], s_t.shape[2])  #1*20*40*4
        
        initial_state = s_t 

        if observe :
            OBSERVE = 999999999    #We keep observe, never train
            epsilon = self.FINAL_EPSILON
            print ("Now we load weight")
            model.load_weights("model.h5")
            adam = Adam(lr=self.LEARNING_RATE)
            model.compile(loss='mse',optimizer=adam)
            print ("Weight load successfully")    
        else:                       #We go to training mode
            OBSERVE = self.OBSERVATION
            epsilon = self._utils.load_obj("epsilon") 
            model.load_weights("model.h5")
            adam = Adam(lr=self.LEARNING_RATE)
            model.compile(loss='mse',optimizer=adam)

        t = self._utils.load_obj("time") # resume from the previous time step stored in file system
        while (True): #endless running
            
            loss = 0
            Q_sa = 0
            action_index = 0
            r_t = 0 #reward at 4
            a_t = np.zeros([self.ACTIONS]) # action at t
            
            #choose an action epsilon greedy
            if t % self.FRAME_PER_ACTION == 0: #parameter to skip frames for actions
                if  random.random() <= epsilon: #randomly explore an action
                    print("----------Random Action----------")
                    action_index = random.randrange(self.ACTIONS)
                    a_t[action_index] = 1
                else: # predict the output
                    q = model.predict(s_t)       #input a stack of 4 images, get the prediction
                    max_Q = np.argmax(q)         # chosing index with maximum q value
                    action_index = max_Q 
                    a_t[action_index] = 1        # o=> do nothing, 1=> jump
                    
            #We reduced the epsilon (exploration parameter) gradually
            if epsilon > self.FINAL_EPSILON and t > OBSERVE:
                epsilon -= (self.INITIAL_EPSILON - self.FINAL_EPSILON) / self.EXPLORE 

            #run the selected action and observed next state and reward
            x_t1, r_t, terminal = game_state.get_state(a_t)
            print('fps: {0}'.format(1 / (time.time()-last_time))) # helpful for measuring frame rate
            last_time = time.time()
            x_t1 = x_t1.reshape(1, x_t1.shape[0], x_t1.shape[1], 1) #1x20x40x1
            s_t1 = np.append(x_t1, s_t[:, :, :, :3], axis=3) # append the new image to input stack and remove the first one
            
            
            # store the transition in D
            D.append((s_t, action_index, r_t, s_t1, terminal))
            if len(D) > self.REPLAY_MEMORY:
                D.popleft()

            #only train if done observing
            if t > OBSERVE: 
                
                #sample a minibatch to train on
                minibatch = random.sample(D, self.BATCH)
                inputs = np.zeros((self.BATCH, s_t.shape[1], s_t.shape[2], s_t.shape[3]))   #32, 20, 40, 4
                targets = np.zeros((inputs.shape[0], self.ACTIONS))                         #32, 2

                #Now we do the experience replay
                for i in range(0, len(minibatch)):
                    state_t = minibatch[i][0]    # 4D stack of images
                    action_t = minibatch[i][1]   #This is action index
                    reward_t = minibatch[i][2]   #reward at state_t due to action_t
                    state_t1 = minibatch[i][3]   #next state
                    terminal = minibatch[i][4]   #wheather the agent died or survided due the action
                    

                    inputs[i:i + 1] = state_t    

                    targets[i] = model.predict(state_t)  # predicted q values
                    Q_sa = model.predict(state_t1)      #predict q values for next step
                    
                    if terminal:
                        targets[i, action_t] = reward_t # if terminated, only equals reward
                    else:
                        targets[i, action_t] = reward_t + self.GAMMA * np.max(Q_sa)

                loss += model.train_on_batch(inputs, targets)
                self.loss_df.loc[len(self.loss_df)] = loss
                self.q_values_df.loc[len(self.q_values_df)] = np.max(Q_sa)
            s_t = initial_state if terminal else s_t1 #reset game to initial frame if terminate
            t = t + 1
            
            # save progress every 1000 iterations
            if t % 1000 == 0:
                print("Now we save model")
                game_state._game.pause() #pause game while saving to filesystem
                model.save_weights("model.h5", overwrite=True)
                self._utils.save_obj(D,"D") #saving episodes
                self._utils.save_obj(t,"time") #caching time steps
                self._utils.save_obj(epsilon,"epsilon") #cache epsilon to avoid repeated randomness in actions
                self.loss_df.to_csv("./objects/loss_df.csv",index=False)
                self.scores_df.to_csv("./objects/scores_df.csv",index=False)
                self.actions_df.to_csv("./objects/actions_df.csv",index=False)
                self.q_values_df.to_csv(self.q_value_file_path,index=False)
                with open("model.json", "w") as outfile:
                    json.dump(model.to_json(), outfile)
                clear_output()
                game_state._game.resume()
            # print info
            state = ""
            if t <= OBSERVE:
                state = "observe"
            elif t > OBSERVE and t <= OBSERVE + self.EXPLORE:
                state = "explore"
            else:
                state = "train"

            print("TIMESTEP", t, "/ STATE", state,             "/ EPSILON", epsilon, "/ ACTION", action_index, "/ REWARD", r_t,             "/ Q_MAX " , np.max(Q_sa), "/ Loss ", loss)

        print("Episode finished!")
        print("************************")
#main function
def playGame(observe=False):
    action = Action()
    dino = Agent(action)
    game_state = Game(dino,action) 
    game_state.init_cache()   
    model = game_state.buildmodel()
    try:
        game_state.trainNetwork(model,game_state,observe=observe)
    except StopIteration:
        action.end()
playGame(observe=True);