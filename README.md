# DDPG Inverted Pendulum Java

This project implements the Deep Deterministic Policy Gradient (DDPG) algorithm to solve the Inverted Pendulum problem. The pendulum is simulated, and the algorithm is trained to balance the pendulum in the upright position.

## Key Variables

Here are the main configurable variables in the code that control the behavior of the simulation and the training process:

- `train` (boolean):  
  Set this to `true` to enable training. Once the training is complete, set it to `false` to stop training and begin using the trained model.

- `Length_Of_Pendulum` (int):  
  The length of the pendulum in meters. Default value is `1`.

- `Mass_Of_Pendulum` (double):  
  The mass of the pendulum in kilograms. Default value is `0.1`.

- `g` (double):  
  The acceleration due to gravity (m/s²). Default value is `10`.

- `initial_position_of_pendulum` (double):  
  The initial position of the pendulum in radians. A value of `0` means the pendulum is upright (balanced), `Math.PI` means fully inverted, and `±Math.PI/2` means horizontal.

- `initial_angular_velocity_of_pendulum` (double):  
  The initial angular velocity of the pendulum (radians per second). Default value is `0`.

- `maxTorque` (int):  
  The action space, representing the maximum torque applied to the pendulum. Larger values make balancing the pendulum more difficult. Default value is `2`.

- `fps` (int):  
  Frames per second for the pendulum animation. Default value is `30`.

- `M` (int):  
  The number of times the agent interacts with the environment to create the `trainingSet.csv` file. Default value is `200`.

- `maxEpisodeRewardRequired` (int):  
  The reward threshold that, when achieved, will stop the training and save the neural network weights. Default value is `-150`.

- `EPISODES` (int):  
  The number of data points in `trainingSet.csv` is calculated as `EPISODES / 0.05`. Default value is `20`.

- `discount` (double):  
  The discount factor for future rewards. Default value is `0.99`.

## Training Process

1. **Enable Training:**
   To start the training process, set the `train` variable to `true` in the code.

   ```java
   static boolean train = true;
   
2. **Hyperparameter Tuning:**
   The `DeepDeterministicPolicyGradient` function can be used to tune various hyperparameters such as the learning rate, activation functions, the number of layers in the neural networks, and more. Below are some 
   of the hyperparameters you can modify in this function:
   
   - **Learning Rate:**
     ```java
     double learn_rate_actor = Math.pow(10, -4);  // Actor network learning rate
     double learn_rate_critic = Math.pow(10, -4);  // Critic network learning rate

   - **Number of neurons in a layer and its activation function:**
     ```java
     for (int j1 = 0; j1 < 16; j1++) {
             critic_L1_Layer.add(new Neuron()
                    .setLearning_Rate(learn_rate_critic)
                    .setSeed(seed)
                    .setActivation(Neuron.Activation_Function.ReLU)
                    .setWeightsFileName("critic")
                    .setWeightsFileIndex(j1)
                    .setWeightInitializationAlgorithm(Neuron.Weight_Initialization.Xavier)
                    .setOptimizerAlgorithm(Neuron.Optimization_Algorithm.Adam)
                    .build());
     }
           

4. **Weights Storage:**
   The weights of the neural networks are saved in the src/DDPG folder after the training process completes. The training process will automatically stop once the `maxEpisodeRewardRequired` is attained
