# F1TENTH: An Over-taking Algorithm Using Machine Learning


> This repository is part of the paper ***"[F1TENTH](https://f1tenth.org/index.html): An Over-taking Algorithm Using Machine Learning"***  
> The model, in the simulator repository, can be only used in the simulator, because not test on a real car yet.   
> If you want to apply this overtaking algorithm to a real F1tenth car, you have to minimize the gap between real car and simulated car first, then collecting some data and training a new model. More detail can be found below and wiki page in the simulator repository. Be prepared for facing a lot of extra work that maybe not relate to the algorithm, such as configuring signal between hardware and software.

Please feel free to raise topics in Issues section, I will try my best to answer them!


## Datasets


### How to create this?


> Assuming the simulator is running well on your computer, and familiar with the basic usages first

**Make sure the parameters of vehicle model won't be changed during creating dataset!!!**

I implemented a feature that if you press a key on keyboard or a button on controller, the program will start recording data, and press again will stop recording data and save data as csv files. Which key or button can be found in params.yaml file.  

1. Let's setup the leading vehicle first. You can use any algorithm you like to drive it, such as MPC, RRT, follow the gap, etc., but I implemented Model Predictive Control as controller. If you want to use MPC, the pipeline of how to generate minimum time trajectory waypoints can be found in [another repository](https://github.com/JZ76/Racetrack-Preparation). There are a lot of parameters in MPC algorithm, where can have huge effects on behaviour of driving, it will take sometime to fine-tune those parameters. I listed some of my experience of how to set parameters in the wiki page of simulator repository.  
2. The ego vehicle can be controlled by a human driver, highly recommend using a Xbox controller, and drive it just like playing a racing game.   
3. Before start creating dataset, it is **NECESSARY** to be familiar with *the feeling of the car*  
4. You can change initial position of two cars in simulator.cpp file    
5. When start recording, the leading vehicle must shown up in ego vehicle's LiDAR, you should see a little square in the simulator. And better start recording with both car is moving rather than static. You should stop recording data when finish overtaking i.e. you cannot see the leading vehicle in LiDAR   
6. During recording, make sure there is no collision with either racetrack or car. If there is collision, you can press Ctrl+C to kill the simulator, so the data won't be saved, or you can save the data and delete them in folder. It would be easier if you sort your dataset in added time order, the last three files always the latest csv files.
7. Try to overtake with different strategies, sometimes overtake from left side, sometimes from right side.

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


As you can see, there are two files in this repo, one is called Colab_version.ipynb, this is for training on Colab or on Windows machine,   
another is called AppleSilicon_version.py, this is for training on M1/M1PRO/M1MAX/M1Ultra machine (probably works on M2/M2PRO?/M2MAX?/M2Ultra? machine).   

You don't have to install all packages as exact the same version as I listed, try to run the code first, see if there is any error that related to package version.  

> [Due to SimpleRNN cannot use cuda cores](https://www.tensorflow.org/guide/keras/rnn#performance_optimization_and_cudnn_kernels), it may take hours to train a model

- Colab/Windows
  - numpy==1.18.5
  - tensorflow==2.8.2
  - pandas==1.4.2

> Due to [some unknown bugs](https://github.com/tensorflow/tensorflow/issues/56082) and incompatible features in tensorflow, I changed some code to make it working on a Apple silicon

- Apple Silicon
  - numpy==1.21.6
  - tensorflow-macos==2.8.0
  - pandas==1.4.2


### Overtaking Algorithm Design Principles (my appoarch)


First of all, the dataset should include overtaking maneuvers in different scenario, i.e. different overtaking strategies in different racetracks. The way of creating models is called *imitation learning* , i.e. a machine learning model will try to copy behaviour from a dataset. Hence, the quality of a dataset determines the performance of a model. 

There are three hypothesis:   
- the leading vehicle will use minimum time trajectory
- the max speed of leading vehicle is lower than the ego vehicle
- the leading vehicle cannot defence overtaking, because the LiDAR sensor has 270° field of view rather than 360°, the car doesn't know what happened in the back

These three hyothesis define boundary of the problem, the leading car playing their best but the top speed is lower than the overtaking car, so that the overtaking car can have opportunities to overtake, and [the competition rules](https://icra2022-race.f1tenth.org/rules.html) said penalty will be applied if upon crashing the opponent.  

After we have a dataset, we need to think about what kind of neural network is more suitable and how do we feed data into the neural network. Since this is imitation learning, we can start thinking by how do we drive a F1tenth car. Imagine you are sitting in front of screen, stare at the car in the simulator, hold a Xbox controller, if this is the first time you drive it, you will build a mapping relationship in your brain which from joystick and trigger to the movement of the car. Usually you will have a good understanding of how to drive well after few attempts. But how did you know the movement of the car? The only way is through LiDAR data, if the LiDAR pattern rotated fast, means the car is rotating fast; if the LiDAR pattern went back fast, means the car is going forward fast. Here, I said we actually combined LiDAR data from past few timestamps together **implicitly**, it is impossible to know the movement of the car from just one LiDAR data. This is the reason why Dense model doesn't work well, and this is where **Recurrent Neural Network** kicks in. RNN is very suitable for process sequence data, such as language, sounds, video. Although it seems like I should use computer vision algorithm because this is how we process it, numerical data here is more precise than image data.  

Is that it? However, there is one failure case, a long straight equal-width racetrack. From the LiDAR data only, you can only know whether the car is moving or not in horizontal direction, but you don't know the speed in vertical direction (along the racetrack). But you still know the car is going forward, why? Because your finger tells you that I am holding the trigger right now, the car should have speed. Hence, we need another input data for neural network: current car speed and steering angle. Note that the real speed and steering angle is different from driving command.   

Now, I need to think about how to glue different layers together to maximize the performance. There are a lot of models in RNN family, such as SimpleRNN, LSTM, GRU, Transformer. LSTM and GRU is designed for remembering elements at the beginning of a very long sequence. When you try to overtake the opponent, you won't remember LiDAR data from 10 seconds ago, because they are useless for current decision making. So, I use the SimpleRNN as one of the layers.   
For rest of layers, I took a inspiration from natural language process, added a Dense layer as [Embedding layer](https://www.youtube.com/watch?v=OuNH5kT-aD0) before the SimpleRNN layer. This Dense layer will filter useless information in LiDAR data, and reinforce important information, such as too close to the racetrack, the position of the component, etc. After the SimpleRNN layer, there are 4 Dense layers as final decision making, making decision by a fusion of the SimpleRNN output and real car speed and steering angle.  

Finally, have a look at the structure   
<img src="https://user-images.githubusercontent.com/6621970/176226712-c1320830-9047-4ebf-9452-025f5a5d0466.jpeg" width="500" height="700">


### Fine-tuning parameters


You can change number of nodes in each layer, and their activation function, as well as the loss function. It may take a while to find a good one that can perform very well on all dataset. You can select a fairly easy racetrack first, like Australia dataset, and create dozens of models, and test them on the same racetrack to see whether the model has good performance or not. Next, you can use a more difficult racetrack, like Shanghai dataset, some of models probably cannot drive well on that, then pick out good ones. And using a easier racetrack again, this is to avoid overfitting problem, one easy racetrack, one hard racetrack. In the end, maybe only very few models can pass all datasets, testing and evaluating them on racetracks that never seen before.  
I put some very good models on the simulator repository, welcome to have a try!


## Results (these racetracks are not in the training dataset)


## Self-driving


![Screen Recording 2022-06-29 at 19 01 18 2](https://user-images.githubusercontent.com/6621970/176650707-4c99e84f-39ce-4f33-bf10-d1f0119ba28c.gif)


## Overtaking


![Screen Recording 2022-06-30 at 11 43 32](https://user-images.githubusercontent.com/6621970/176659102-7a56f1a0-d849-4b13-9b5c-ddbde6702443.gif)


## Following


![Screen Recording 2022-06-30 at 11 15 37](https://user-images.githubusercontent.com/6621970/176659173-0bedccf9-b67d-4744-97b6-b2b426f37d4e.gif)


## Static obstacle avoiding


![Screen Recording 2022-06-30 at 11 55 51](https://user-images.githubusercontent.com/6621970/176660843-ec40cb38-78bb-4308-abd4-b1289012397d.gif)
