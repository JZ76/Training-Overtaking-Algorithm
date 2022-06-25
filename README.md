# [DISSERTATION] Training-Overtaking-Algorithm


> This repository is the second part of my dissertation ***"[F1TENTH](https://f1tenth.org/index.html): Development of A Multi-Agent Simulator and An Overtaking Algorithm Using Machine Learning"***  

Please feel free to raise topics in Issues section, I will try my best to answer them!


## Datasets


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
