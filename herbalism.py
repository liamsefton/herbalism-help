from tkinter import *
from PIL import ImageGrab
from PIL import ImageTk,Image 
from cv2 import dnn_superres
import cv2 as cv
import pygame
import numpy as np
import time
from multiprocessing import Process, freeze_support
from screeninfo import get_monitors
import traceback

calibrating = False

def start_calibration():
    global calibrating
    calibrating = True

def take_screenshot(x1, y1, x2, y2, ss_num):
    image_name = "ss" + str(ss_num) + ".jpg"
    #Edit this so that the screenshots grab only your minimap, bbox takes arguments X1, Y1 for top left corner of square and X2, Y2 for bottom right corner. Might be monitor resolution dependent.
    screenshot = ImageGrab.grab(bbox=(x1, y1, x2, y2))
    screenshot.save(image_name, "JPEG")

def main():
    global calibrating

    first_pass = True

    print("Scanning for herbs...")

    threshold = 0.8
    print("Threshold is 80%")

    x1 = float(get_monitors()[0].width) * 0.85
    y1 = 0
    x2 = float(get_monitors()[0].width)
    y2 = float(get_monitors()[0].height) * 0.23

    sr = dnn_superres.DnnSuperResImpl_create()
    sr.readModel("FSRCNN_x2.pb")
    sr.setModel("edsr", 3)

    root = Tk()  
    root.title("HerbCam")

    herb_icon_filename = "herb1.jpg"

    if(float(get_monitors()[0].height) == 1080):
        canvas = Canvas(root, width = int((x2-x1)*1.75), height = int((y2-y1)*1.75))  
        #canvas = Canvas(root)
    else:
        #canvas = Canvas(root, width = (x2-x1), height = (y2-y1))
        canvas = Canvas(root, width = int((x2-x1)), height = int((y2-y1))) 
        #canvas = Canvas(root) 

    needle = cv.imread(herb_icon_filename, cv.IMREAD_UNCHANGED)

    needle = cv.cvtColor(needle, cv.COLOR_RGB2GRAY)
    needle = sr.upsample(needle.astype(np.float32))
    needle = cv.cvtColor(needle, cv.COLOR_GRAY2RGB)
    needle_orig = needle.copy()

    canvas.pack() 

    calibrate_button = Button(root, text="Calibrate", command=start_calibration)
    calibrate_button.pack()

    w1 = Scale(root, from_=0, to=100, orient=HORIZONTAL, label="Volume", length=300)
    w1.pack()

    w2 = Scale(root, from_=0, to=100, orient=HORIZONTAL, label="Threshold in %", length=300)
    w2.pack()

    w1.set(50)
    w2.set(80)
    volume = w1.get() / 100
    threshold = w2.get() / 100

    pygame.init()
    sound = pygame.mixer.Sound("notification.wav")
    sound.set_volume(volume)

    time_counter = 0
    ss_counter = 0
    min_val, max_val, min_loc, max_loc = 0, 0, 0, 0

    canvas_img_orig = ""

    try:
        while 1==1:
            ss_num = ss_counter % 4
            ss_filename = "ss" + str(ss_num) + ".jpg"
            if time_counter % 32 == 0:
                ss_counter += 1
                take_screenshot(x1, y1, x2, y2, ss_num)

                #Polling rate. If you want faster polling rate, lower float value.
                time.sleep(0.1)

                haystack = cv.imread(ss_filename, cv.IMREAD_UNCHANGED)
                if(float(get_monitors()[0].height) == 1080):
                    haystack = cv.resize(haystack, (0, 0), fx=1.75, fy=1.75)

                haystack = cv.cvtColor(haystack, cv.COLOR_RGB2GRAY)
                haystack = sr.upsample(haystack.astype(np.float32))
                haystack = cv.cvtColor(haystack, cv.COLOR_GRAY2RGB)

                #herb_img = ImageTk.PhotoImage(Image.fromarray(cv.resize(cv.cvtColor(sr.upsample(cv.cvtColor(haystack, cv.COLOR_RGB2GRAY).astype(np.float32)), cv.COLOR_GRAY2RGB), (0, 0), fx=.5, fy=.5)))
                herb_img = ImageTk.PhotoImage(Image.fromarray(cv.resize(haystack, (0, 0), fx=.5, fy=.5)))

                if first_pass:
                    canvas_img_orig = canvas.create_image(20, 20, anchor=NW, image=herb_img)
                else:
                    canvas.itemconfig(canvas_img_orig, image=herb_img)
                
                result = cv.matchTemplate(haystack.astype(np.uint8), needle.astype(np.uint8), cv.TM_CCOEFF_NORMED) #cv.TM_CCORR_NORMED, mask=transparent_mask)

                min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)

            if calibrating:
                haystack = cv.imread(ss_filename, cv.IMREAD_UNCHANGED)
                if(float(get_monitors()[0].height) == 1080):
                    haystack = cv.resize(haystack, (0, 0), fx=1.75, fy=1.75)

                
                haystack = cv.cvtColor(haystack, cv.COLOR_RGB2GRAY)
                haystack = sr.upsample(haystack.astype(np.float32))
                haystack = cv.cvtColor(haystack, cv.COLOR_GRAY2RGB)

                result_max = 0
                scale = 1


                print("Calibrating...")
                for img in range(4):
                    herb_icon_filename = "herb" + str(img+1) + ".jpg"
                    needle_new = cv.imread(herb_icon_filename, cv.IMREAD_UNCHANGED)

                    needle_new = cv.cvtColor(needle_new, cv.COLOR_RGB2GRAY)
                    needle_new = sr.upsample(needle_new.astype(np.float32))
                    needle_new = cv.cvtColor(needle_new, cv.COLOR_GRAY2RGB)
                    needle_orig = needle_new.copy()

                    for i in range(100):
                        avg = 0
                        temp_needle = ""
                        for j in range(5):
                            temp_needle = cv.resize(needle_orig, (0, 0), fx=(i+50)/100, fy=(i+50)/100)
                            temp_needle = cv.cvtColor(temp_needle, cv.COLOR_RGB2GRAY)
                            temp_needle = sr.upsample(temp_needle.astype(np.float32))
                            temp_needle = cv.cvtColor(temp_needle, cv.COLOR_GRAY2RGB)
                            temp_needle = cv.resize(temp_needle, (0, 0), fx=0.5, fy=0.5)
                            
                            result = cv.matchTemplate(haystack.astype(np.uint8), temp_needle.astype(np.uint8), cv.TM_CCOEFF_NORMED)
                            min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
                            avg += max_val
                        
                        avg = avg / 5
                        if avg > result_max:
                            scale = (i+50)/100
                            print("Calibration: " + str(avg) + " for scale " + str(scale) + " with image " + herb_icon_filename)
                            result_max = avg
                            needle = temp_needle
                            #needle = cv.cvtColor(needle, cv.COLOR_RGB2GRAY)
                            #needle = sr.upsample(needle.astype(np.float32))
                            #needle = cv.resize(needle, (0, 0), fx=0.5, fy=0.5)
                            #needle = cv.cvtColor(needle, cv.COLOR_GRAY2RGB)
                calibrating = False
                print("Done calibrating.")
            
            root.update()

            percent_val = max_val*100
            if time_counter % 128 == 0:
                print(str(percent_val) + "% confidence level.")
                if max_val >= threshold:
                    #Download any .wav to customize notification.
                    sound.play()

            #Polling rate.
            #time.sleep(0.05)
            time_counter += 1
            volume = w1.get() / 100
            threshold = w2.get() / 100
            sound.set_volume(volume)
            first_pass = False

    except Exception as Argument:
        f = open("error_log" + time.strftime("%Y%m%d-%H%M%S") + ".txt", "w")
        f.write(str(Argument))
        f.write(traceback.format_exc())
        f.close()
        

if __name__ == "__main__":
    freeze_support()
    Process(target=main).start()

