from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
import time

class DinoGame:
    def __init__(self):
        driver_path = "/home/galaksiya/ai_project/selenium-driver/chromedriver"
        chrome_options = Options()
        chrome_options.add_argument("disable-infobars")
        chrome_options.add_argument("--mute-audio")
        self.browser = webdriver.Chrome(executable_path = driver_path,chrome_options=chrome_options)
        self.browser.set_window_position(x=-10,y=0)
        self.browser.get('chrome://dino')
        self.browser.execute_script("Runner.config.ACCELERATION=0")
        init_script = "document.getElementsByClassName('runner-canvas')[0].id = 'runner-canvas'"
        self.browser.execute_script(init_script)
        #self.browser.implicitly_wait(30)
        #self.browser.maximize_window()


    def get_crashed(self):
        return self.browser.execute_script("return Runner.instance_.crashed")
    def get_playing(self):
        return self.browser.execute_script("return Runner.instance_.playing")
    def restart(self):
        self.browser.execute_script("Runner.instance_.restart()")
    def press_up(self):
        self.browser.find_element_by_tag_name("body").send_keys(Keys.ARROW_UP)
    def press_down(self):
        self.browser.find_element_by_tag_name("body").send_keys(Keys.ARROW_DOWN) 
    def press_right(self):
        self.browser.find_element_by_tag_name("body").send_keys(Keys.ARROW_RIGHT)

    def get_score(self):
        score_array = self.browser.execute_script("return Runner.instance_.distanceMeter.digits")
        score = ''.join(score_array)
        return int(score)

    def get_highscore(self):
        score_array = self.browser.execute_script("return Runner.instance_.distanceMeter.highScore")
        for i in range(len(score_array)):
            if score_array[i] == '':
                break
        score_array = score_array[i:]        
        score = ''.join(score_array)
        return int(score)

    def pause(self):
        return self.browser.execute_script("return Runner.instance_.stop()")
    def resume(self):
        return self.browser.execute_script("return Runner.instance_.play()")
    def end(self):
        self.browser.close()

if __name__ == "__main__":
  dino =  DinoGame()
  dino.press_up()