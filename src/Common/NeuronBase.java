package src.Common;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public abstract class NeuronBase {

    public Double feedForward(List<NeuronBase> L_prev, List<Double> Input) throws IOException{return 0.0;};

    public void feedBack(double error){};

    public void Memorize(){};

    public List<Double> feedForwardSoftMax(List<NeuronBase> L_prev, List<Double> Input) throws IOException{return new ArrayList<>();};

    public void feedBackSoftMax( int j, List<Double> Output){};

    public boolean isResidual(){return false;};

    public NeuronBase build(){return this;};

    public NeuronBase setActivation(Neuron.Activation_Function activationFunction){
        return this;
    }

    public NeuronBase setWeightsFileName(String memoryFileIndex){
        return this;
    }

    public NeuronBase setWeightsFileIndex(int memoryFileIndex){
        return this;
    }

    public NeuronBase setSeed(long seed){
        return this;
    }

    public NeuronBase setOptimizerAlgorithm(Neuron.Optimization_Algorithm optimization_algorithm){
        return this;
    }

    public NeuronBase setWeightInitializationAlgorithm(Neuron.Weight_Initialization weight_initialization){
        return this;
    }

    public NeuronBase setLearning_Rate(double learning_rate){
        return this;
    }


    public void feedForward(List<NeuronBase> L_prev, int batchIndex) throws IOException{};

    public void feedForward(List<NeuronBase> L_prev) throws IOException{};

    public double getOutput(){
        return -99;
    };

    public double getGradient(int batchIndex){
        return -99;
    };

    public void setInput(double input){};

    public void setError_Next(double error_Next){};

    public void feedBack(int batchIndex, int epoch){};

    public void feedBackGradient(int batchIndex, int epoch){};

    public void feedBackSoftMax(int batchIndex, double[] Y_Actual){};

    public double[] getSoftMaxOutput(){return new double[]{};};

    public double[] getWeights(){return new double[]{};}

    public void setWeights(double[] weights){}

    public void setWeightsPolyak(double[] weights){}

    public void reInitializeBatchMemory(){}

    public void feedBackSoftMax(int batchIndex, double[] Y_Actual, double advantage){};

    public String getWeightsFileName(){
        return "this";
    }
}
