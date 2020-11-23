from subprocess import Popen, PIPE, STDOUT
import shlex
import time
import sys
import psutil
import os, signal 

mod = 'aa' # first good moving camera data generated with carla version 0.97
mod = 'test' # 100 views 
#mod = 'ab' # 88 views, yaw 0 to 359nes - 5 sceenes test
mod = 'ac' # 51 views, yaw 0 to 359 - 10 scenes
mod = 'ad' # 51 views, yaw 0 to 359 - 100 scenes
mod = 'ae' # 51 views, yaw 0 to 359 - 10 scenes
mod = 'af' # 51 views, yaw 0 to 359 - 10 scenes
mod = 'ag' # 51 views, yaw 0 to 359 - 10 scenes
mod = 'test' # 51 views, yaw 0 to 359 - 10 scenes
mod = 'ah' # 51 views, yaw 0 to 359 - 5 scenes
mod = 'ai' # 51 views, yaw 0 to 359 - 10 scenes
mod = 'aj' # 51 views, yaw 0 to 359 - 10 scenes
mod = 'ak' # 51 views, yaw 0 to 359 - 300 scenes
mod = 'test' # 51 views, yaw 0 to 359 - 10 scenes

#mod = 'rotaa'#43 views, 40 vehicles, 
mod = 'rot_ab'#43 views, 35 vehicles, 
mod = 'rot_ac'#43 views, 35 vehicles, 

mod = 'test' # 51 views, yaw 0 to 359 - 10 scenes


# mod = 'ep_aa' # 51 views, yaw 0 to 359 - 10 scenes

mod = 'hiaa'
mod = 'hiab'
mod = 'hiac'
mod = 'hiad'
mod = 'hiae'
mod = 'hiaf'
mod = 'hiag'
mod = 'hiah'
mod = 'hiai'
mod = 'hiaj'
mod = 'hiak'
mod = 'hial'
mod = 'hiam'
mod = 'hian'
mod = 'hiao'
mod = 'hiap' # The above are all 43 views, 30 vehicles

mod = 'test'

mod = 'vehaa' # two vehicles, rotate around the two vehicles, second veh atleast 5 meters away
mod = 'vehab' # no bikes, two vehicles, rotate around the two vehicles, second veh atleast 5 meters away
mod = 'test'
mod = 'mr06' # with segmentation masks


save_dir = '/hdd/gsarch/data'

carla_sim = "/hdd/carla97/CarlaUE4.sh -carla-server -windows -ResX=100 -ResY=100 -benchmark"
carla_sim_args = shlex.split(carla_sim)
cnt = 0
for i in range(0,100):
    p1 = Popen(carla_sim_args, stdout=PIPE, stderr=PIPE)
    time.sleep(10)
    print("Number of times carla simulator started: ", cnt)
    cnt+=1
    p2 = Popen(["python3.5","new_static_cam_2veh_sem.py", str(i), mod, save_dir], stdout=PIPE, stderr=PIPE)
    time.sleep(1)

    out, err = p2.communicate()
    print(out.decode('utf-8'))
    print(err.decode('utf-8'))
    # for line in out.decode("utf-8").split('\\n'):
    #     print('\t' + line)
    # print('ERROR')
    # for line in err.decode("utf-8").split('\\n'):
    #     print('\t' + line)
    
    p1.terminate()
    time.sleep(5)
    # Iterate over all running process
    for proc in psutil.process_iter():
        try:
            # Get process name & pid from process object.
            processName = proc.name()
            processID = proc.pid
            #print(processName , ' ::: ', processID)
            if 'Carla' in processName:
                print("PROCESS FOUND")
                print(processName)
                os.kill(processID, signal.SIGSTOP) 
                print("PROCESS STOPPED")
                time.sleep(5)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    for proc in psutil.process_iter():
        try:
            # Get process name & pid from process object.
            processName = proc.name()
            processID = proc.pid
            #print(processName , ' ::: ', processID)
            if 'Carla' in processName:
                print("PROCESS FOUND")
                print(processName)
                os.kill(processID, signal.SIGKILL) 
                print("PROCESS KILLED")
                time.sleep(5)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    print("Done with single iteration. Terminating everything")
    print("==========================================================")
    p2.terminate()
    time.sleep(10)
