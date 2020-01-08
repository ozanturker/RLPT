class Agent:
    def __init__(self,action): #takes game as input for taking actions
        self._action = action; 
        self.jump(); #to start the game, we need to jump once
    def is_running(self):
        return self._action.get_playing()
    def is_crashed(self):
        return self._action.get_crashed()
    def jump(self):
        self._action.press_up()
    def duck(self):
        self._action.press_down()