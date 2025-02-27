package src.DDPG;

import src.Common.InputNeuron;
import src.Common.Neuron;
import src.Common.NeuronBase;

import javax.swing.*;
import java.io.*;
import java.util.*;

public class DDPG {
    static boolean train = false;
    static int Length_Of_Pendulum = 1;//0.326;
    static double Mass_Of_Pendulum = 0.1;
    static double g = 10; // acceleration due to gravity
    static double initial_position_of_pendulum = Math.PI; // 0 - upright(balanced), Math.PI = fully inverted, +-Math.PI/2 = horizontal
    static double initial_angular_velocity_of_pendulum = 0;
    public static int maxTorque = 2; // action space. Heavier the pendulum more the torque needed to balance(difficult to train)
    static int fps = 30; // frames per second for pendulum animation
    public static String Filepath = "./src/DDPG/"; // path to store weights of Neural Networks

    int M = 200; // number of times to interact with environment to create trainingSet.csv. Critic and Actor networks are trained alternatively.
    int maxEpisodeRewardRequired = -150; // the reward, which on attaining, training stops and weights of Neural networks are stored
    int EPISODES = 20; // EPISODES/0.05 = Number of data points in trainingSet.csv
    int EPOCH = 1;
    double discount = 0.99;

    double [] Policy_Loss_Batch = new double[]{};
    double [] Critic_Loss_Batch = new double[]{};
    boolean trainCritic = true;
    boolean trainActor = true;

    double yi;

    List<NeuronBase> policy_InputLayer = new ArrayList<>();
    List<NeuronBase> policy_L1_Layer = new ArrayList<>();
    List<NeuronBase> policy_L2_Layer = new ArrayList<>();
    List<NeuronBase> policy_Output_Layer = new ArrayList<>();

    List<NeuronBase> policy_target_InputLayer = new ArrayList<>();
    List<NeuronBase> policy_target_L1_Layer = new ArrayList<>();
    List<NeuronBase> policy_target_L2_Layer = new ArrayList<>();
    List<NeuronBase> policy_target_Output_Layer = new ArrayList<>();

    List<NeuronBase> critic_InputLayer = new ArrayList<>();
    List<NeuronBase> critic_L1_Layer = new ArrayList<>();
    List<NeuronBase> critic_L2_Layer = new ArrayList<>();
    List<NeuronBase> critic_Output_Layer = new ArrayList<>();

    List<NeuronBase> target_InputLayer = new ArrayList<>();
    List<NeuronBase> target_L1_Layer = new ArrayList<>();
    List<NeuronBase> target_L2_Layer = new ArrayList<>();
    List<NeuronBase> target_Output_Layer = new ArrayList<>();

    public static void main(String[] args) throws Exception{
        if (args.length > 0) {
            // Convert the argument to boolean
            boolean flag = Boolean.parseBoolean(args[0]);
            if (flag) {
                train = true;
            }
        }
        DDPG ddpg = new DDPG();
        ddpg.DeepDeterministicPolicyGradient();
    }

    public void DeepDeterministicPolicyGradient() throws IOException, InterruptedException {
        List<Double> avgRewardList = new ArrayList<>();
        double learn_rate_actor =  Math.pow(10,-4);
        double learn_rate_critic = Math.pow(10,-4);
        long seed = -98;
        // Initialize policy network
        policy_Output_Layer.add(new Neuron()
                .setLearning_Rate(learn_rate_actor)
                .setSeed(seed)
                .setActivation(Neuron.Activation_Function.tanH)
                .setWeightsFileName("actor")
                .setWeightsFileIndex(policy_L2_Layer.size() + policy_L1_Layer.size())
                .setWeightInitializationAlgorithm(Neuron.Weight_Initialization.Xavier)
                .setOptimizerAlgorithm(Neuron.Optimization_Algorithm.Adam)
                .build());

        // Initialize policy target network
        policy_target_Output_Layer.add(new Neuron()
                .setLearning_Rate(learn_rate_actor)
                .setSeed(seed)
                .setActivation(Neuron.Activation_Function.tanH)
                .setWeightsFileName("targetActor")
                .setWeightsFileIndex(policy_target_L2_Layer.size() + policy_target_L1_Layer.size())
                .setWeightInitializationAlgorithm(Neuron.Weight_Initialization.Xavier)
                .setOptimizerAlgorithm(Neuron.Optimization_Algorithm.Adam)
                .build());


        // Initialise critic
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

        for (int j1 = 0; j1 < 16; j1++) {
            critic_L2_Layer.add(new Neuron()
                    .setLearning_Rate(learn_rate_critic)
                    .setSeed(seed)
                    .setActivation(Neuron.Activation_Function.ReLU)
                    .setWeightsFileName("critic")
                    .setWeightsFileIndex(j1 + critic_L1_Layer.size())
                    .setWeightInitializationAlgorithm(Neuron.Weight_Initialization.Xavier)
                    .setOptimizerAlgorithm(Neuron.Optimization_Algorithm.Adam)
                    .build());
        }

        for (int j1 = 0; j1 < 1; j1++) {
            critic_Output_Layer.add(new Neuron()
                    .setLearning_Rate(learn_rate_critic)
                    .setSeed(seed)
                    .setActivation(Neuron.Activation_Function.LINEAR)
                    .setWeightsFileName("critic")
                    .setWeightsFileIndex(j1 + critic_L1_Layer.size() + critic_L2_Layer.size())
                    .setWeightInitializationAlgorithm(Neuron.Weight_Initialization.Xavier)
                    .setOptimizerAlgorithm(Neuron.Optimization_Algorithm.Adam)
                    .build());
        }

        // init  target critic
        for (int j1 = 0; j1 < 16; j1++) {
            target_L1_Layer.add(new Neuron()
                    .setLearning_Rate(learn_rate_critic)
                    .setSeed(seed)
                    .setActivation(Neuron.Activation_Function.ReLU)
                    .setWeightsFileName("targetCritic")
                    .setWeightsFileIndex(j1)
                    .setWeightInitializationAlgorithm(Neuron.Weight_Initialization.Xavier)
                    .setOptimizerAlgorithm(Neuron.Optimization_Algorithm.Adam)
                    .build());
        }

        for (int j1 = 0; j1 < 16; j1++) {
            target_L2_Layer.add(new Neuron()
                    .setLearning_Rate(learn_rate_critic)
                    .setSeed(seed)
                    .setActivation(Neuron.Activation_Function.ReLU)
                    .setWeightsFileName("targetCritic")
                    .setWeightsFileIndex(j1 + target_L1_Layer.size())
                    .setWeightInitializationAlgorithm(Neuron.Weight_Initialization.Xavier)
                    .setOptimizerAlgorithm(Neuron.Optimization_Algorithm.Adam)
                    .build());
        }

        for (int j1 = 0; j1 < 1; j1++) {
            target_Output_Layer.add(new Neuron()
                    .setLearning_Rate(learn_rate_critic)
                    .setSeed(seed)
                    .setActivation(Neuron.Activation_Function.LINEAR)
                    .setWeightsFileName("targetCritic")
                    .setWeightsFileIndex(j1 + target_L1_Layer.size() + target_L2_Layer.size())
                    .setWeightInitializationAlgorithm(Neuron.Weight_Initialization.Xavier)
                    .setOptimizerAlgorithm(Neuron.Optimization_Algorithm.Adam)
                    .build());
        }

        for(int v=0 ; v < M ; v++) {
            if( train) {
                // Swap between training actor and critic for each time training data is created from environment
                if(  (v+1) % 2 == 0) {
                    boolean swap = trainActor;
                    trainActor = trainCritic;
                    trainCritic = swap;
                }

                File trainingDataFile = new File(Filepath + "trainingData.csv");
                if (trainingDataFile.exists()) {
                    boolean delete = trainingDataFile.delete();
                    //System.out.println("trainingDataFile.delete() - " + v + " " + delete);
                }
                double episodeReward = 0;
                // createTrainingData by interacting with the Environment for EPISODES
                Environment_Pendulum environment = new Environment_Pendulum(Math.PI,0);
                Environment_Pendulum.State curr_state = environment.getState();
                Environment_Pendulum.State new_state;

                for (double i = 0; i < EPISODES; i = i + 0.05) {
                    List<Double> curr_state_NN_Input = currStateToInput(curr_state);
                    feedForward_Policy_Network(curr_state_NN_Input, 0);
                    double at = policy_Output_Layer.get(0).getOutput();

                    // convert to force
                    double force = sampleAction(at);

                    new_state = environment.getNewStateAndReward(force);
                    double reward = new_state.reward;
                    episodeReward = episodeReward + reward;

                    List<Double> new_state_policy_target_Input = currStateToInput(new_state);

                    StoreTrainingData(curr_state_NN_Input, at, reward, new_state_policy_target_Input);

                    curr_state = new_state;
                    clearCache();
                }
                //episodeReward = episodeReward/EPISODES ;
                //System.out.println("episodeReward == " + episodeReward);
                avgRewardList.add(episodeReward);
                // if (episodeReward > maxEpisodeRewardRequired) break;
                List<List<String>> trainingSet = new ArrayList<>();
                try (BufferedReader br = new BufferedReader(new FileReader(Filepath + "trainingData.csv"))) {
                    String line;
                    while ((line = br.readLine()) != null) {
                        String[] values = line.split(",");
                        trainingSet.add(Arrays.asList(values));
                    }
                }

                Neuron.BATCH = 64;//trainingSet.size();
                //BATCH = trainingSet.size()/1000;
                reInitBatchMemory();
                Policy_Loss_Batch = new double[Neuron.BATCH];
                Critic_Loss_Batch = new double[Neuron.BATCH];

                //standardize(Reward_Function);

                //Collections.shuffle(trainingSet);
                int epoch = 0;
                // Train critic/actor using training data
                while (train && (epoch++ < EPOCH)) {
                    Collections.shuffle(trainingSet);
                    List<List<String>> batch = new ArrayList<>();
                    for (int index = 0; index < trainingSet.size(); index++) {
                        batch.add(trainingSet.get(index));
                        if( (index != trainingSet.size()-1) && ((index + 1) % Neuron.BATCH != 0)) continue;

                        if( trainCritic) {
                            // Calculate Critic Loss
                            for (int batchIndex = 0; batchIndex < batch.size(); batchIndex++) {
                                try {
                                    List<String> data = batch.get(batchIndex);//trainingSet.get(index);

                                    //int batchIndex = index % Neuron_DDPG.BATCH;

                                    double reward = Double.parseDouble(data.get(2));
                                    double action = Double.parseDouble(data.get(1));
                                    List<Double> s = stringToList(data.get(0));
                                    List<Double> s_1 = stringToList(data.get(3));

                                    feedForward_Policy_Target_Network(s_1, batchIndex);
                                    double at_1 = policy_target_Output_Layer.get(0).getOutput();

                                    List<Double> s_1_copy_1 = deepCopyList(s_1);
                                    feedForward_Critic_Target_Network(s_1_copy_1, batchIndex, new ActionInputNeuron(sampleAction(at_1))); //
                                    double Q_at_1 = target_Output_Layer.get(0).getOutput();

                                    if (reward == 0) {
                                        System.out.println("TERMINOLO !!! ");
                                        yi = reward;
                                    } else
                                        yi = ((reward) + discount * Q_at_1);

                                    List<Double> s_copy = deepCopyList(s);
                                    ActionInputNeuron actionInputNeuron = new ActionInputNeuron(sampleAction(action));
                                    feedForward_Critic_Network(s_copy, batchIndex, actionInputNeuron);
                                    double Q_action = critic_Output_Layer.get(0).getOutput();
                                    Critic_Loss_Batch[batchIndex] = (yi - Q_action);
                                } catch (Exception e) {
                                    System.out.println("Exception " + e);
                                }
                            }
                            // Train Critic Network to make Q(s,a) closer to value wanted by Bellman(Gradient Descent)
                            for (int batchIndex = 0; batchIndex < batch.size(); batchIndex++) {
                                feedback_Critic_Network(batchIndex, Critic_Loss_Batch[batchIndex] / batch.size(), 0);
                            }
                        }

                        if( trainActor) {
                            // Calculate Actor Loss
                            for (int batchIndex = 0; batchIndex < batch.size(); batchIndex++) {
                                List<String> data = batch.get(batchIndex);
                                List<Double> sfb = stringToList(data.get(0));

                                feedForward_Policy_Network(sfb, batchIndex);
                                double u_sfb = policy_Output_Layer.get(0).getOutput();

                                List<Double> s_copy_3 = deepCopyList(sfb);
                                ActionInputNeuron actionInputNeuron = new ActionInputNeuron(sampleAction(u_sfb));
                                feedForward_Critic_Network(s_copy_3, batchIndex, actionInputNeuron);
                                double Q_u_sfb = critic_Output_Layer.get(0).getOutput();

                                // Backpropagate the gradients of critic network to be used to train actor network
                                Backpropagate_Gradients_Critic_Network(batchIndex, 1, 0);
                                double df_Q_u_sfb = actionInputNeuron.getGradient(batchIndex);

                                Policy_Loss_Batch[batchIndex] = -df_Q_u_sfb;

                                clearCache();
                            }
                            // Once Critic is trained, find action for which Q(s,a) is maximum by training Actor Network (Gradient Ascent)
                            for (int batchIndex = 0; batchIndex < batch.size(); batchIndex++) {
                                feedback_Policy_Network(batchIndex, Policy_Loss_Batch[batchIndex] / batch.size(), 0);
                            }
                        }
                        copyWeights();
                        clearCache();
                        batch.clear();
                    }
                    if( trainCritic) {
                        System.out.println("actorLoss == " + (findMean(Policy_Loss_Batch, Policy_Loss_Batch.length)));
                    }
                }
            }
            double avgEpisodeReward = 0;
            avgEpisodeReward = (findMean(avgRewardList, 4));
            if( v > 4) {
                System.out.println("avgEpisodeReward == " + avgEpisodeReward);
                if (avgEpisodeReward > maxEpisodeRewardRequired) break;
            }
        }

        // Store Neural Network weights
        if( train) {
            File statsFile = new File(Filepath + "actor.csv");
            if (statsFile.exists()) {
                System.out.println("statsFile.delete() - " + statsFile.delete());
            }
            File statsFile2 = new File(Filepath + "targetActor.csv");
            if (statsFile2.exists()) {
                System.out.println("statsFile2.delete() - " + statsFile2.delete());
            }
            File statsFile3 = new File(Filepath +"critic.csv");
            if (statsFile3.exists()) {
                System.out.println("statsFile3.delete() - " + statsFile3.delete());
            }

            File statsFile4 = new File(Filepath +"targetCritic.csv");
            if (statsFile4.exists()) {
                System.out.println("statsFile4.delete() - " + statsFile4.delete());
            }
            Store_ANN_Weights();
        }

        // Testing
        {
            Environment_Pendulum environment = new Environment_Pendulum(initial_position_of_pendulum,initial_angular_velocity_of_pendulum);
            Environment_Pendulum.State curr_state = environment.getState();
            Environment_Pendulum.State new_state ;

            JFrame f = new JFrame("Pendulum2");
            Pendulum2 p = new Pendulum2(Length_Of_Pendulum);
            p.setState(curr_state);
            f.add(p);
            f.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
            f.pack();
            f.setVisible(true);
            new Thread(p).start();

            for (double i = 0; i < 60; i = i + 0.05) {
                System.out.println("state: " +curr_state.toString());

                List<Double> curr_state_NN_Input = currStateToInput(curr_state);//stateSpaceToInput(stateSpaceQuantization);
                feedForward_Policy_Network(curr_state_NN_Input,0);

                double at = policy_Output_Layer.get(0).getOutput();

                // convert to force
                double force = sampleAction(at);
                System.out.println("action: " +force);
                new_state = environment.getNewStateAndReward(force);
                curr_state = new_state;
                p.setState(curr_state);

                clearCache();
                Thread.sleep(1000/fps);
            }
        }
        Thread.sleep(2000);
    }

    // Forward Propagation of Critic's Q Network
    public void feedForward_Critic_Network(List<Double> input, int batchIndex, NeuronBase action) throws IOException {
        clearCache();
        boolean dense = true;

        for (Double d : input) {
            critic_InputLayer.add(new InputNeuron(d));
        }
        List<NeuronBase> prev_Layer = critic_InputLayer;

        if (!critic_L1_Layer.isEmpty()){
            for (int i =0; i<critic_L1_Layer.size();i++) {
                if( !dense) {
                    List<NeuronBase> prev_Layer1 = new ArrayList<>();
                    prev_Layer1.add(prev_Layer.get(i%prev_Layer.size()));
                    critic_L1_Layer.get(i).feedForward(prev_Layer1,batchIndex );
                }
                else
                    critic_L1_Layer.get(i).feedForward( prev_Layer,batchIndex );
            }
            prev_Layer= critic_L1_Layer;
        }

        List<NeuronBase> l2_input = new ArrayList<>();
        l2_input.add(action);
        prev_Layer = l2_input;

        if (!critic_L2_Layer.isEmpty()){
            for (NeuronBase neuron : critic_L2_Layer) {
                neuron.feedForward(prev_Layer,batchIndex );
            }
            prev_Layer= critic_L2_Layer;
        }

        List<NeuronBase> concat = new ArrayList<>(critic_L1_Layer);
        concat.addAll(critic_L2_Layer);
        prev_Layer = concat;

        critic_Output_Layer.get(0).feedForward(prev_Layer,batchIndex );
        //clearCache();
    }

    // Forward Propagation of Critic's Target Network
    public void feedForward_Critic_Target_Network(List<Double> input, int batchIndex, NeuronBase action) throws IOException {
        clearCache();
        boolean dense = true;

        //target_InputLayer.add(action);
        for (Double d : input) {
            target_InputLayer.add(new InputNeuron(d));
        }
        List<NeuronBase> prev_Layer = target_InputLayer;
        if (target_L1_Layer.size() > 0){
            for (int i =0; i<target_L1_Layer.size();i++) {
                if( !dense) {
                    List<NeuronBase> prev_Layer1 = new ArrayList<>();
                    prev_Layer1.add(prev_Layer.get(i%prev_Layer.size()));
                    target_L1_Layer.get(i).feedForward(prev_Layer1,batchIndex );
                }
                else
                    target_L1_Layer.get(i).feedForward(prev_Layer,batchIndex );
            }
            prev_Layer= target_L1_Layer;
        }

        List<NeuronBase> l2_input = new ArrayList<>();
        l2_input.add(action);
        prev_Layer = l2_input;

        if (target_L2_Layer.size() > 0){
            for (NeuronBase neuron : target_L2_Layer) {
                neuron.feedForward(prev_Layer,batchIndex );
            }
            prev_Layer= target_L2_Layer;
        }

        List<NeuronBase> concat = new ArrayList<>(target_L1_Layer);
        concat.addAll(target_L2_Layer);
        prev_Layer = concat;

        target_Output_Layer.get(0).feedForward(prev_Layer,batchIndex );
    }

    // Forward Propagation of Actor's Target Network
    public void feedForward_Policy_Target_Network(List<Double> input, int batchIndex) throws IOException {
        clearCache();
        boolean dense = true;
        for (Double d : input) {
            policy_target_InputLayer.add(new InputNeuron(d));
        }
        List<NeuronBase> prev_Layer = policy_target_InputLayer;
        if (!policy_target_L1_Layer.isEmpty()){
            for (int i =0; i<policy_target_L1_Layer.size();i++) {
                if( !dense) {
                    List<NeuronBase> prev_Layer1 = new ArrayList<>();
                    prev_Layer1.add(prev_Layer.get(i%prev_Layer.size()));
                    policy_target_L1_Layer.get(i).feedForward(prev_Layer1, batchIndex);
                }
                else
                    policy_target_L1_Layer.get(i).feedForward(prev_Layer , batchIndex);
            }
            prev_Layer= policy_target_L1_Layer;
        }


        if (!policy_target_L2_Layer.isEmpty()){
            for (NeuronBase neuron : policy_target_L2_Layer) {
                neuron.feedForward(prev_Layer,batchIndex );
            }
            prev_Layer= policy_target_L2_Layer;
        }

        policy_target_Output_Layer.get(0).feedForward(prev_Layer,batchIndex );
    }

    // Forward Propagation of Actor's Q Network
    public void feedForward_Policy_Network(List<Double> input, int batchIndex) throws IOException {
        clearCache();
        boolean dense = true;
        for (Double d : input) {
            policy_InputLayer.add(new InputNeuron(d));
        }
        List<NeuronBase> prev_Layer = policy_InputLayer;
        if (!policy_L1_Layer.isEmpty()){
            for (int i =0; i<policy_L1_Layer.size();i++) {
                if( !dense) {
                    List<NeuronBase> prev_Layer1 = new ArrayList<>();
                    prev_Layer1.add(prev_Layer.get(i%prev_Layer.size()));
                    policy_L1_Layer.get(i).feedForward( prev_Layer1, batchIndex);
                }
                else
                    policy_L1_Layer.get(i).feedForward( prev_Layer, batchIndex);
            }
            prev_Layer= policy_L1_Layer;
        }


        if (!policy_L2_Layer.isEmpty()){
            for (NeuronBase neuron : policy_L2_Layer) {
                neuron.feedForward(prev_Layer,batchIndex );
            }
            prev_Layer= policy_L2_Layer;
        }

        policy_Output_Layer.get(0).feedForward(prev_Layer,batchIndex );
    }

    public void feedback_Policy_Network(int batchIndex, double advantage, int epoch){
        policy_Output_Layer.get(0).setError_Next(advantage);

        if (!policy_Output_Layer.isEmpty()) {
            for (NeuronBase neuronBase : policy_Output_Layer) {
                neuronBase.feedBack(batchIndex, epoch);
            }
        }

        if (!policy_L2_Layer.isEmpty()) {
            for (NeuronBase neuronBase : policy_L2_Layer) {
                neuronBase.feedBack(batchIndex, epoch);
            }
        }

        if (!policy_L1_Layer.isEmpty()) {
            for (NeuronBase neuronBase : policy_L1_Layer) {
                neuronBase.feedBack(batchIndex, epoch);
            }
        }
    }

    public void feedback_Critic_Network(int batchIndex, double error, int epoch) throws InterruptedException {

        critic_Output_Layer.get(0).setError_Next(error);

        if (!critic_Output_Layer.isEmpty()) {
            for (NeuronBase neuronBase : critic_Output_Layer) {
                neuronBase.feedBack(batchIndex, epoch);
            }
        }

        if (!critic_L2_Layer.isEmpty()) {
            for (NeuronBase neuronBase : critic_L2_Layer) {
                neuronBase.feedBack(batchIndex, epoch);
            }
        }

        if (!critic_L1_Layer.isEmpty()) {
            for (NeuronBase neuronBase : critic_L1_Layer) {
                neuronBase.feedBack(batchIndex, epoch);
            }
        }
    }

    public void Backpropagate_Gradients_Critic_Network(int batchIndex, double error, int epoch){

        critic_Output_Layer.get(0).setError_Next(error);

        if (!critic_Output_Layer.isEmpty()) {
            for (NeuronBase neuronBase : critic_Output_Layer) {
                neuronBase.feedBackGradient(batchIndex, epoch);
            }
        }

        if (!critic_L2_Layer.isEmpty()) {
            for (NeuronBase neuronBase : critic_L2_Layer) {
                neuronBase.feedBackGradient(batchIndex, epoch);
            }
        }

        if (!critic_L1_Layer.isEmpty()) {
            for (NeuronBase neuronBase : critic_L1_Layer) {
                neuronBase.feedBackGradient(batchIndex, epoch);
            }
        }
    }

    public void clearCache(){
        policy_InputLayer.clear();
        policy_target_InputLayer.clear();
        critic_InputLayer.clear();
        target_InputLayer.clear();
    }

    public List<Double> currStateToInput(Environment_Pendulum.State state){
        List<Double> input = new ArrayList<>();
        input.add(state.x);
        input.add(state.y);
        input.add(state.theta_dot);
        return input;
    }

    public double clip( double tau, double maxMag){
        if ( tau > maxMag)return maxMag;
        else return Math.max(tau, (-1 * maxMag));
    }

    public void copyWeights(){
        for (int i=0 ; i< critic_L1_Layer.size(); i ++){
            target_L1_Layer.get(i).setWeightsPolyak(critic_L1_Layer.get(i).getWeights());
        }

        for (int i=0 ; i< critic_L2_Layer.size(); i ++){
            target_L2_Layer.get(i).setWeightsPolyak(critic_L2_Layer.get(i).getWeights());
        }

        for (int i=0 ; i< critic_Output_Layer.size(); i ++){
            target_Output_Layer.get(i).setWeightsPolyak(critic_Output_Layer.get(i).getWeights());
        }

        for (int i=0 ; i< policy_L1_Layer.size(); i ++){
            policy_target_L1_Layer.get(i).setWeightsPolyak(policy_L1_Layer.get(i).getWeights());
        }

        for (int i=0 ; i< policy_L2_Layer.size(); i ++){
            policy_target_L2_Layer.get(i).setWeightsPolyak(policy_L2_Layer.get(i).getWeights());
        }

        for (int i=0 ; i< policy_Output_Layer.size(); i ++){
            policy_target_Output_Layer.get(i).setWeightsPolyak(policy_Output_Layer.get(i).getWeights());
        }
    }

    public double sampleAction(double at){
/*        if( at> 0)
            return at * (2);
        else return at * (-2);*/
        //return clip(at  , Environment2.actionSpace);
        return at * maxTorque;
    }

    public void reInitBatchMemory(){
        if (!policy_L1_Layer.isEmpty()){
            for (NeuronBase neuron : policy_L1_Layer) {
                neuron.reInitializeBatchMemory();
            }
        }

        if (!policy_L2_Layer.isEmpty()){
            for (NeuronBase neuron : policy_L2_Layer) {
                neuron.reInitializeBatchMemory();
            }
        }

        if (!policy_Output_Layer.isEmpty()) {
            for (NeuronBase neuron : policy_Output_Layer) {
                neuron.reInitializeBatchMemory();
            }
        }


        if (!policy_target_L1_Layer.isEmpty()){
            for (NeuronBase neuron : policy_target_L1_Layer) {
                neuron.reInitializeBatchMemory();
            }
        }

        if (!policy_target_L2_Layer.isEmpty()){
            for (NeuronBase neuron : policy_target_L2_Layer) {
                neuron.reInitializeBatchMemory();
            }
        }

        if (!policy_target_Output_Layer.isEmpty()) {
            for (NeuronBase neuron : policy_target_Output_Layer) {
                neuron.reInitializeBatchMemory();
            }
        }


        if (!critic_L1_Layer.isEmpty()){
            for (NeuronBase neuron : critic_L1_Layer) {
                neuron.reInitializeBatchMemory();
            }
        }

        if (!critic_L2_Layer.isEmpty()){
            for (NeuronBase neuron : critic_L2_Layer) {
                neuron.reInitializeBatchMemory();
            }
        }

        if (!critic_Output_Layer.isEmpty()) {
            for (NeuronBase neuron : critic_Output_Layer) {
                neuron.reInitializeBatchMemory();
            }
        }

        if (!target_L1_Layer.isEmpty()){
            for (NeuronBase neuron : target_L1_Layer) {
                neuron.reInitializeBatchMemory();
            }
        }

        if (!target_L2_Layer.isEmpty()){
            for (NeuronBase neuron : target_L2_Layer) {
                neuron.reInitializeBatchMemory();
            }
        }

        if (!target_Output_Layer.isEmpty()) {
            for (NeuronBase neuron : target_Output_Layer) {
                neuron.reInitializeBatchMemory();
            }
        }
    }

    public double findMean (List<Double> X, int n) {
        double sum =0;
        double mean =0;
        // calculate mean and std of all columns
        for (int i = X.size()-1, k = 0; k < n && i>-1; i--, k++) {
            sum = (sum + X.get(i));
        }

        mean = sum/n;
        return mean;
    }

    public double findMean (double []X, int n) {
        double sum =0;
        double mean =0;
        // calculate mean and std of all columns
        for (int i = X.length-1, k = 0; k < n && i>-1; i--, k++) {
            sum = (sum + X[i]);
        }

        mean = sum/n;
        return mean;
    }

    public List<Double> stringToList(String str){
        List<Double> integers = new ArrayList<>();
        String[] strings = str.split(";");
        for (String string : strings) {
            integers.add(Double.parseDouble(string));
        }
        return integers;
    }

    // Store Weights of Neural Networks into CSV files
    public void Store_ANN_Weights(){
        for (NeuronBase neuronL4 : policy_L1_Layer) {
            neuronL4.Memorize();
        }

        for (NeuronBase neuronL4 : policy_L2_Layer) {
            neuronL4.Memorize();
        }

        for (NeuronBase neuronL4 : policy_Output_Layer) {
            neuronL4.Memorize();
        }

        for (NeuronBase neuronL4 : policy_target_L1_Layer) {
            neuronL4.Memorize();
        }

        for (NeuronBase neuronL4 : policy_target_L2_Layer) {
            neuronL4.Memorize();
        }

        for (NeuronBase neuronL4 : policy_target_Output_Layer) {
            neuronL4.Memorize();
        }

        for (NeuronBase neuronL4 : critic_L1_Layer) {
            neuronL4.Memorize();
        }

        for (NeuronBase neuronL4 : critic_L2_Layer) {
            neuronL4.Memorize();
        }

        for (NeuronBase neuronL4 : critic_Output_Layer) {
            neuronL4.Memorize();
        }

        for (NeuronBase neuronL4 : target_L1_Layer) {
            neuronL4.Memorize();
        }

        for (NeuronBase neuronL4 : target_L2_Layer) {
            neuronL4.Memorize();
        }

        for (NeuronBase neuronL4 : target_Output_Layer) {
            neuronL4.Memorize();
        }
    }

    // Store training data after interacting with Environment
    public void StoreTrainingData(List<Double> s_t, double at, double reward, List<Double> s_tNext){
        try {
            File statsFile = new File(Filepath + "trainingData.csv");
            if (statsFile.exists()) {
            } else {
                FileWriter out = new FileWriter(statsFile);
                out.flush();
                out.close();
            }

            if (statsFile.exists()) {
                FileWriter buf = new FileWriter(Filepath+ "trainingData.csv", true);
                for (Double integer : s_t) {
                    buf.append(String.valueOf(integer));
                    buf.append(";");
                }
                buf.append(",");
                buf.append(String.valueOf(at));
                buf.append(",");
                buf.append(String.valueOf(reward));
                buf.append(",");
                for (Double integer : s_tNext) {
                    buf.append(String.valueOf(integer));
                    buf.append(";");
                }
                buf.append("\n");
                buf.flush();
                buf.close();
            } else {
                System.out.println("StoreTrainingData FAIL 3 NO FILE");
            }
        }
        catch (Exception ex){
            System.out.println("StoreTrainingData FAIL 4"  + ex.getMessage());
        }
    }

    public void normalise(double[] L2inputs){
        double min = 0;
        double max = 0;
        for (double l2input : L2inputs) {
            if (l2input < min)
                min = l2input;
            if (l2input > max)
                max = l2input;
        }
        // find min-max
        if ( max!= min) {
            for (int i = 0; i < L2inputs.length; i++) {
                L2inputs[i] = Math.abs(((L2inputs[i] - min) / (max - min)));
            }
        }
    }

    public List<Double> deepCopyList(List<Double> oldList){
        return new ArrayList<>(oldList);
    }
}
