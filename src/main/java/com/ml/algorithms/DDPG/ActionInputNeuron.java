package src.main.java.com.ml.algorithms.DDPG;

import src.main.java.com.ml.algorithms.Common.NeuronBase;

public class ActionInputNeuron extends NeuronBase {

    public double input;
    public double gradient;

    public ActionInputNeuron(double input) {
        this.input = input;
    }

    @Override
    public void setInput(double input){
        this.input = input;
    }

    @Override
    public double getOutput(){
       return input;
    }

    public void setError_Next(double error_Next){
        gradient = gradient + error_Next;
    }

    @Override
    public double getGradient(int batchIndex){
        return gradient;
    }
}
