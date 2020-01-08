import cv2 
import pickle
import numpy as np
from PIL import Image
from io import BytesIO
import base64
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
import time
class Utils:
    
    def __init__(self):
        self.getbase64Script = "canvasRunner = document.getElementById('runner-canvas'); \
        return canvasRunner.toDataURL().substring(22)"

    def save_obj(self,obj, name ):
        with open('objects/'+ name + '.pkl', 'wb') as f: #dump files into objects folder
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    
    def load_obj(self,name ):
        with open('objects/' + name + '.pkl', 'rb') as f:
            return pickle.load(f)

    def grab_screen(self,_driver):
        image_b64 = _driver.execute_script(self.getbase64Script)
        screen = np.array(Image.open(BytesIO(base64.b64decode(image_b64))))
        image = self.process_img(screen)#processing image as required
        return image

    def process_img(self, image):
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #RGB to Grey Scale
        image = image[:300, :500] #Crop Region of Interest(ROI)
        image = cv2.resize(image, (80,80))
        return  image

    def show_img(self, graphs = False):
        """
        Show images in new window
        """
        while True:
            screen = (yield)
            window_title = "logs" if graphs else "game_play"
            cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)        
            imS = cv2.resize(screen, (800, 400)) 
            cv2.imshow(window_title, screen)
            if (cv2.waitKey(1) & 0xFF == ord('q')):
                cv2.destroyAllWindows()
                break
if __name__ == '__main__':
    driver_path = "/home/galaksiya/ai_project/selenium-driver/chromedriver"
    chrome_options = Options()
    chrome_options.add_argument("disable-infobars")
    chrome_options.add_argument("--mute-audio")
    browser = webdriver.Chrome(executable_path = driver_path,chrome_options=chrome_options)
    browser.set_window_position(x=-10,y=0)
    browser.get('chrome://dino')
    browser.execute_script("Runner.config.ACCELERATION=0")
    init_script = "document.getElementsByClassName('runner-canvas')[0].id = 'runner-canvas'"
    browser.execute_script(init_script)    
    image = Utils().grab_screen(browser)
    Utils().show_img()
    print(image)
    