# [DISSERTATION] Training-Overtaking-Algorithm


> This repository is the second part of my dissertation ***"[F1TENTH](https://f1tenth.org/index.html): Development of A Multi-Agent Simulator and An Overtaking Algorithm Using Machine Learning"***  
> The model, in the simulator repository, can be only used in the simulator, because not test on a real car yet.   
> If you want to apply this overtaking algorithm to a real F1tenth car, you have to minimize the gap between real car and simulated car first, then collecting some data and training a new model. More detail can be found below and wiki page in the simulator repository. Be prepared for facing a lot of extra work that maybe not relate to the algorithm, such as configuring signal between hardware and software.

Please feel free to raise topics in Issues section, I will try my best to answer them!


## Datasets


### How to create this?


> Assuming the simulator is running well on your computer

**Make sure the parameters of vehicle model won't be changed during creating dataset!!!**

I implemented a feature that if you press a key on keyboard or a button on controller, the program will start recording data, and press again will stop recording data and save data as csv files. Which key or button can be found in params.yaml file.  

1. Let's setup the leading vehicle first. You can use any algorithm you like to drive it, such as MPC, RRT, follow the gap, etc., but I implemented Model Predictive Control as controller. If you want to use MPC, the pipeline of how to generate minimum time trajectory waypoints can be found in another repository. There are a lot of parameters in MPC algorithm, where can have huge effects on behaviour of driving, it will take sometime to fine-tune those parameters. I listed some of my experience of how to set parameters in the wiki page of simulator repository.  
2. The ego vehicle can be controlled by a human driver, highly recommend using a Xbox controller, and drive it just like playing a racing game.   
3. Before start creating dataset, it is **NECESSARY** to be familiar with *the feeling of the car*  
4. You can change initial position of two cars in simulator.cpp file    
5. When start recording, the leading vehicle must shown up in ego vehicle's LiDAR, you should see a little square in the simulator. And better start recording with both car is moving rather than static. You should stop recording data when finish overtaking i.e. you cannot see the leading vehicle in LiDAR   
6. During recording, make sure there is no collision with either racetrack or car. If there is collision, you can press Ctrl+C to kill the simulator, so the data won't be saved, or you can save the data and delete them in folder. It would be easier if you sort your dataset in added time order, the last three files always the latest csv files.
7. Try to overtake with different strategies, like sometimes overtake from left side, sometimes from right side.

### Dataset Info


| Name | Size | Number of Files |
| :--- | --- | :---: |
| [Australia racetrack dataset](https://drive.google.com/drive/folders/15dC_8rJeR8NdOGdYL9UcydJbIoYOcNKv?usp=sharing)  | 3.16 GB | 408 |
| [Shanghai racetrack dataset](https://drive.google.com/drive/folders/1LTy3OdV9xb5wfOrsg5az78shDPH4PdlK?usp=sharing) | 4.13 GB | 579 |
| [Gulf racetrack dataset](https://drive.google.com/drive/folders/1n7G9ZCKGhQVXsBCFU-rgMqVMhx6Wda88?usp=sharing) | 2.56 GB | 201 |
| [Malaysian racetrack dataset](https://drive.google.com/drive/folders/1BNZFJpgWyqisEXyiZrBeysqKMa03_5Eu?usp=sharing) | 3.51 GB | 576 |


### Data Preview


car_state_blue_XXXX or car_state_red_XXXX  

> Sidenote: vehicle model is bicycle model with dynamic equations.  
> blue car is the car perform overtaking behaviour, aka ego vehicle.  
> red car is the leading vehicle.  


| Position_X | Position_Y | Theta | Velocity_X | Velocity_Y | Steering_angle | Angular_velocity | slip_angle |
| :---: | :----: | :---: | :---: | :---: | :---: | :---: | :---: |
| x value of position coordinate in world frame of reference | y value of position coordinate in world frame of reference | vehicle heading direction in world frame of reference, 0 is the positive direction of X axis, and rotate toward positive of Y axis is increase theta | velocity in x axis direction but under vehicle frame of reference | velocity in y axis direction but under vehicle frame of reference | steering angle of the front wheels | angular velocity of center point of mass | slip angle of front wheels |
| meter | meter | radians | m/s | m/s | radians | dθ/dt | radians |


ML_dataset_blueXXXX

> Speed and steering angle are the driving command, the real speed and steer angle are not the same as driving command  
> the vehicle model determines the mapping between driving command and real value   

| Speed | Steering_angle | LiDAR_scan |
| :---: | :---: | :---: |
| desired speed | desired steering angle | raw LiDAR data (simulated) |
| m/s | radians | meter |


## Training A Model


### Environment


As you can see, there are two files in this repo, one is called Colab_version.ipynb, this is for training on Colab or on Windows machine, another is called AppleSilicon_version.py, this is for training on M1/M1PRO/M1MAX/M1Ultra machine (probably works on M2/M2PRO?/M2MAX?/M2Ultra? machine).   

You don't have to install all packages as exact the same version as I listed, try to run the code first, see if there is any error that related to package version.  

- Colab/Windows
  - numpy==1.18.5
  - tensorflow==2.8.2
  - pandas==1.4.2

- Apple Silicon
  - numpy==1.21.6
  - tensorflow-macos==2.8.0
  - pandas==1.4.2


### Model Design Principles


We aim at developing a model that can overtake an opponent car, the dataset should include overtaking maneuvers in different scenario, i.e. different overtaking strategies in different racetracks. 


There are two hypothesis:   
- the leading vehicle will use minimum time trajectory
- the leading vehicle cannot defence overtaking, because the LiDAR sensor has 270° field of view rather than 360°, the car doesn't know what happened in the back

You can select a fairly easy racetrack first, and test your model on same racetrack to see whether the model has good performance or not
