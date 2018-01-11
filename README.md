# Udacity_Self_Driving_Car-Behavioural_Cloning-P3

Project: Behavioural Cloning.
In this project I have built and trained deep neural network to enable a car to drive itself around the racing track in a driving simulator. The starting model is a convolutional neural network based on the [NVDIA](https://arxiv.org/pdf/1704.07911.pdf) architecture.

The project is organised by the following sections:
* Loading Data
* Data Preprocessing
* Design and Test Model Architecture
* Model Improvement
* Testing and Results

## Project structure
|      File       |               Description                                                                                            |
|---------------- |----------------------------------------------------------------------------------------------------------------------|
|   'model.py'    | main script, implements and trains deep neural network for end-to-end driving.                                       |
|   'model.h5'    | model weights                                                                                                        |
|   'drive.py'    | implements driving simulator callbaks, communicates with the driving simulator.                                      |

## Loading Data
The simulator has too tracks, one of which was used to collect training data, the other (unseen) track is effectively a substitute for the test set enabling to test model generalization properties (the ability to drive on unseen terrain).

The driving simulator saves images coming from three front-facing (left, central and right) cameras, as well as various driving variables such as steering angle, speed, breaking and throttle. The objective is to predict th steering angle in the range [-1,+1] based on frontal cameras images.

The model is that of ''behavioural cloning'', the deep neural net learns to emulate the behaviour of human drivers by learning the relationship betweeen images coming from the frontal cameras and the steering angle. The model ''learns from data'' by deciding which features of terrain, road markings are important without involvement of human engineers.

![](images/german_signs.png)
