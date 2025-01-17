package src.Common;

public class InputNeuron extends NeuronBase {

    public double input;

    public InputNeuron(double input) {
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
}
