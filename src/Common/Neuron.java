package src.Common;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class Neuron extends NeuronBase {
    public static double polyak = 0.995;
    List<NeuronBase> L_prev = null;
    boolean zScale = false;
    boolean min_max = false;
    boolean batch= false;
    static Random randomGenerator = new Random();
    public static int BATCH = 100;
    public enum Activation_Function { SIGMOID, ReLU, LINEAR, SOFTMAX, CUSTOM, LEAKY_ReLU, tanH};
    // Required
    public double []W = new double[]{};
    public double b;

    public double []vdW = new double[]{};
    public double []sdW = new double[]{};

    public double []W_FeedForward = new double[]{};

    public enum Optimization_Algorithm  {Gradient_Descent, Adam}
    public enum Weight_Initialization  {Xavier, HE, gaussian, constant}
    double learning_rate = 0;
    long seed = -984;
    double initial_weight = 0.001;//(0.006 * new Random().nextDouble()) - 0.003;
    double momentum = 0.5;
    double epsilon = Math.pow(10, -8);
    double beta1 = 0.9;
    double beta2 = 0.999;

    Activation_Function activationFunction;

    // Values to configure
    public static Optimization_Algorithm optimization_algorithm = Optimization_Algorithm.Adam;
    public static Weight_Initialization weight_initialization = Weight_Initialization.HE;
    int memoryFileIndex = 0;
    String memoryFileName = "";
    boolean isResidual = false;

    // Data
    int p; // total number of predictor variables
    double y;
    double error_Next = 0;
    //List<Double> xi;
    public double [] X = new double[]{};

    public double [] X_Scaled = new double[]{};
    public double [] X_BatchNorm = new double[]{};

    double mean = 0; // total number of predictor variables
    double variance = 1; // total number of predictor variables
    public double V = 1;
    public double B = 0;

    public double e = 0.05;
    boolean isMemoryLoaded = false;
    public double dActivationFunction_dt;

    public double [][]X_Batch = new double[BATCH][];

    public double []Gradient_Batch = new double[BATCH];

    public double [][]WFF_Batch = new double[BATCH][];

    public double []Df_Prev = new double[]{};

    public Neuron(){

    }

    public Neuron(double learning_rate, Activation_Function activationFunction, int memoryFileIndex, String memoryFileName, boolean isResidual){
        this.memoryFileIndex = memoryFileIndex;
        this.activationFunction = activationFunction;
        this.isResidual = isResidual;
        this.learning_rate = learning_rate;
        this.memoryFileName = memoryFileName;
    }

    public Neuron(double learning_rate, Activation_Function activationFunction, int memoryFileIndex, String memoryFileName, Weight_Initialization weight_initialization){
        this.memoryFileIndex = memoryFileIndex;
        this.activationFunction = activationFunction;
        this.weight_initialization = weight_initialization;
        this.learning_rate = learning_rate;
        this.memoryFileName = memoryFileName;
    }

    public NeuronBase setWeightsFileName(String memoryFileName){
        this.memoryFileName = memoryFileName;
        return this;
    }

    public String getWeightsFileName(){
        return this.memoryFileName;
    }

    public NeuronBase setWeightsFileIndex(int memoryFileIndex){
        this.memoryFileIndex = memoryFileIndex;
        return this;
    }

    public NeuronBase setActivation(Activation_Function activationFunction){
        this.activationFunction = activationFunction;
        return this;
    }

    public NeuronBase setSeed(long seed){
        randomGenerator = new Random(seed);
        this.seed = seed;
        return this;
    }

    public NeuronBase setOptimizerAlgorithm(Optimization_Algorithm optimization_algorithm){
        this.optimization_algorithm = optimization_algorithm;
        return this;
    }

    public NeuronBase setWeightInitializationAlgorithm(Weight_Initialization weight_initialization){
        this.weight_initialization = weight_initialization;
        return this;
    }

    public NeuronBase setLearning_Rate(double learning_rate){
        this.learning_rate = learning_rate;
        return this;
    }

    public boolean isResidual(){return isResidual;};

    @Override
    public double[] getWeights() {
        return W;
    }

    @Override
    public void setWeightsPolyak(double[] weights) {
        double []A_Copy = new double[weights.length];
        for(int i=0; i< weights.length; i++) {
            A_Copy[i] = (polyak * A_Copy[i]) + ((1 - polyak) * weights[i]);
        }
        W = Matrix.deepCopy(A_Copy);
    }

    @Override
    public void setWeights(double[] weights) {
        W = Matrix.deepCopy(weights);
    }

    public void reInitializeBatchMemory(){
        X_Batch = new double[BATCH][];
        Gradient_Batch = new double[BATCH];
        WFF_Batch = new double[BATCH][];
    }
    public double activationFunction( double t){
        switch (activationFunction){
            case ReLU:{
                if ( t >= 0)
                    return t;
                else return 0;
            }
            case LEAKY_ReLU: {
                if ( t >= 0)
                    return t;
                else return (Math.pow(10,-10) * t);
            }
            case SIGMOID: {
                return (/*Math.exp(t)*/ 1/ (1 + Math.exp(-1*t)));
            }
            case tanH: {
                return ((Math.exp(t) - Math.exp(-1*t)) / (Math.exp(t) + Math.exp(-1*t)));
            }
            case LINEAR:{
                return t;
            }
            case CUSTOM:{
                return (1 / Math.pow(t,2));
            }
        }
        System.out.println("Something is horribly wrong " );
        return 0;
    }

    public double dActivationFunction_dt( double t){
        switch (activationFunction){
            case ReLU:{
                if ( t >= 0)
                    return 1;
                else return 0;
            }
            case LEAKY_ReLU:{
                if ( t >= 0)
                    return 1;
                else return Math.pow(10,-10);
            }
            case SIGMOID: {
                return activationFunction(t) * ( 1 - activationFunction(t));
                //return (Math.exp(-1*t)/Math.pow(1 + Math.exp(-1 * t),2));
            }
            case tanH:{
                return (1 - Math.pow(activationFunction(t),2));
            }
            case LINEAR:{
                return 1;
            }
            case CUSTOM:{
                return (-2 / Math.pow(t,3));
            }
        }
        System.out.println("Something is horribly wrong " );
        return 0;
    }

    public double d2ActivationFunction_dt2( double t){
        switch (activationFunction){
            case ReLU:
            case LEAKY_ReLU:
            case LINEAR: {
                return 0;
            }
            case SIGMOID: {
                return activationFunction(t) * ( 1 - activationFunction(t)) *  ( 1 - 2* activationFunction(t));
                //return (Math.exp(-1*t)/Math.pow(1 + Math.exp(-1 * t),2));
            }
            case tanH:{
                return ( -2 * activationFunction(t) * (1 - Math.pow(activationFunction(t),2)));
            }
            case CUSTOM:{
                return (2 * t);
            }
        }
        System.out.println("Something is horribly wrong " );
        return 0;
    }

    void frameInput(int batchIndex){
        for ( int j = 0 ; j < L_prev.size() ; j++ ){
            X[j] = L_prev.get(j).getOutput();
        }
        X_Batch[batchIndex] = X;
        if( (batch /*&& (variance ==10 && mean == 10)*/ )|| zScale )
            findMeanAndVariance();

        if(batch)
            createXScaledAndXNorm();
        else if (zScale || min_max)
            X = normalise(X);
    }

    @Override
    public void feedForward(List<NeuronBase> L_prev, int batchIndex) {
        this.L_prev = L_prev;
        p = L_prev.size();
        X = new double[p];
        if( batch) {
            X_Scaled = new double[p];
            X_BatchNorm = new double[p];
        }
        frameInput(batchIndex);
        if( !isMemoryLoaded) {
            W = new double[p];
            vdW = new double[p];
            sdW = new double[p];
            Df_Prev = new double[p];
            try {
                List<List<String>> records1 = new ArrayList<>();
                try (BufferedReader br1 = new BufferedReader(new FileReader("src/main/java/com/ml/algorithms/DDPG/" +memoryFileName + ".csv"))) {
                    String line1;
                    while ((line1 = br1.readLine()) != null) {
                        String[] values1 = line1.split(",");
                        records1.add(Arrays.asList(values1));
                    }
                }
                int i = 0;
                for (i = 0; i < records1.get(memoryFileIndex).size() - 1; i++) {
                    W[i] = convertToNumber(records1.get(memoryFileIndex).get(i));
                }
                b = convertToNumber(records1.get(memoryFileIndex).get(i));
            } catch (Exception e1) {
                System.out.println("memoryFileIndex == " + + memoryFileIndex + "   " + e1.getMessage());

                for (int j = 0; j < p; j++) {
                    W[j] = initialize();
                }
               // b = (2 * random.nextDouble()) - 1;
            }
            isMemoryLoaded = true;
        }
        W_FeedForward = Matrix.deepCopy(W);
        WFF_Batch[batchIndex] = W_FeedForward;

        double yiBeforeActivation = Matrix.dot(W_FeedForward,batch ? X_BatchNorm : X) ;//+ b;
        y = activationFunction(yiBeforeActivation);

        dActivationFunction_dt = dActivationFunction_dt(yiBeforeActivation);
        Gradient_Batch[batchIndex] = dActivationFunction_dt;
    }

    public double initialize() {
        double weight = 0;
        if( weight_initialization.equals(Weight_Initialization.Xavier)) {
            double upper = (1 / Math.sqrt(p));
            double lower = -(1 / Math.sqrt(p));
            weight = (randomGenerator.nextDouble() * (upper - lower)) + lower;
        }
        else if( weight_initialization.equals(Weight_Initialization.HE)){
            double std = (Math.sqrt(2.0/p));
            weight = (randomGenerator.nextDouble() * std);
        }
        else if( weight_initialization.equals(Weight_Initialization.gaussian)){
            weight = (randomGenerator.nextGaussian());
        }
        else if( weight_initialization.equals(Weight_Initialization.constant)){
            weight = initial_weight;
        }
        return weight;
    }

    @Override
    public void feedBack(int batchIndex, int epoch) {
        double[] errorPrev = Matrix.scalar( WFF_Batch[batchIndex],Gradient_Batch[batchIndex]);
        for ( int j = 0 ; j < errorPrev.length ; j++ ){
            double ePrev = error_Next * errorPrev[j];
            if(Double.isNaN(ePrev )|| Double.isInfinite(ePrev) )
                ePrev = 0;

            ePrev = (isResidual ? (ePrev + 1) : ePrev);
            if (batch)
                ePrev = UpdateBatchNormalizer(ePrev,j);

            if ( L_prev.size() > 0)
                L_prev.get(j).setError_Next(ePrev);
        }
        UpdateWeights(batchIndex,epoch);
        //UpdateBias(batchIndex);
        error_Next = 0;
    }

    @Override
    public void feedBackGradient(int batchIndex, int epoch) {
        double[] errorPrev = Matrix.scalar( WFF_Batch[batchIndex],Gradient_Batch[batchIndex]);
        for ( int j = 0 ; j < errorPrev.length ; j++ ){
            double ePrev = error_Next * errorPrev[j];
            if(Double.isNaN(ePrev )|| Double.isInfinite(ePrev) )
                ePrev = 0;

            ePrev = (isResidual ? (ePrev + 1) : ePrev);
            if (batch)
                ePrev = UpdateBatchNormalizer(ePrev,j);

            if ( L_prev.size() > 0)
                L_prev.get(j).setError_Next(ePrev);
        }
        error_Next = 0;
    }

    public void UpdateWeights(int batchIndex, int epoch){
        X = X_Batch[batchIndex];
        dActivationFunction_dt = Gradient_Batch[batchIndex];
        for(int j = 0; j< W.length ; j++ ) {
            double df = error_Next * X[j] * (dActivationFunction_dt);

            if( optimization_algorithm.equals(Optimization_Algorithm.Gradient_Descent)) {
                df = (learning_rate * df);

                if (!Double.isNaN((W[j]) + (df)) && !Double.isInfinite((W[j]) + (df))) {
                    W[j] = (W[j]) + (df) + (momentum * Df_Prev[j]);
                }
                Df_Prev[j] = df;
            }
            else if( optimization_algorithm.equals(Optimization_Algorithm.Adam)) {
                vdW[j] = (beta1 * vdW[j]) + ((1 - beta1) * df );
                sdW[j] = (beta2 * sdW[j]) + ((1 - beta2) * Math.pow(df, 2));
                double vdw_corrected = vdW[j] / (1 - Math.pow(beta1, epoch+1));
                double sdw_corrected = sdW[j] / (1 -Math.pow(beta2, epoch+1));
                W[j] = W[j] + learning_rate * (vdw_corrected / (Math.sqrt(sdw_corrected) + epsilon));
            }
        }
    }

    public void UpdateBias(int batchIndex){
        X = X_Batch[batchIndex];
        dActivationFunction_dt = Gradient_Batch[batchIndex];
        double df = (learning_rate  * error_Next * (dActivationFunction_dt ));
        if( !Double.isNaN((b) + (df)) && !Double.isInfinite((b) + (df)))
            b = (b) + (df);
    }

    public double UpdateBatchNormalizer(final double ePrev, int j){
        double dl_dxscaled = ePrev * V;
        double dl_dV = ePrev * X_Scaled[j];
        double dl_dB = ePrev;
        double dl_dvariance = ePrev * ( X[j] - mean) * ((-V/2) * Math.pow(variance + e,-3/2.0));
        double dl_mean = (ePrev * ((-V/Math.sqrt(variance+e)))) + (dl_dvariance * (1.0/X.length) * (-2) * ( X[j] - mean) );

        V = V + learning_rate * dl_dV;
        B = B + learning_rate * dl_dB;
        variance = variance + learning_rate * dl_dvariance;
        mean = mean + learning_rate * dl_mean;

        return (dl_dxscaled * (1/Math.sqrt(variance + e))) + (dl_dvariance * (1.0/X.length) * (2) * ( X[j] - mean))+ (dl_mean * (1.0/X.length));
    }

    public void setError_Next(double error_Next){
        this.error_Next = this.error_Next + error_Next;
        if( Double.isNaN(error_Next))
            System.out.println("Neuron error_Next is NaN !!! " );
        if( Double.isInfinite(error_Next))
            System.out.println("Neuron error_Next isInfinite !!! " );
/*        if( error_Next == 0)
            System.out.println("Neuron error_Next is 0" );*/
    }
    public void Memorize(){
        try {
            File statsFile = new File( "src/main/java/com/ml/algorithms/DDPG/" + memoryFileName+ ".csv");
            if (statsFile.exists()) {
            } else {
                FileWriter out = new FileWriter(statsFile);
                out.flush();
                out.close();
            }

            if (statsFile.exists()) {
                FileWriter buf = new FileWriter("src/main/java/com/ml/algorithms/DDPG/" + memoryFileName+ ".csv", true);
                for (double v1 : W) {
                    buf.append(String.valueOf(v1)).append(",");
                }
                buf.append(String.valueOf(b));
                buf.append("\n");
                buf.flush();
                buf.close();
            } else {
                System.out.println("FILE NOT FOUND");
            }
        }
        catch (Exception ex){
            System.out.println("EXCEPTION"  + ex.getMessage());
        }
    }

    public double convertToNumber(String str){
        try{
            return Double.parseDouble(str);
        }
        catch (NumberFormatException e){
            StringBuilder stringBuilder = new StringBuilder();
            str = str.toUpperCase();
            for(int i=0; i<str.length(); i++)
            {
                int asciiValue = str.charAt(i);
                stringBuilder.append(asciiValue);
            }
            return Double.parseDouble(stringBuilder.toString());
        }
    }

    public void findMeanAndVariance () {
        double sum =0;
        double sum2 =0;
        // calculate mean and std of all columns
        for (int i = 0; i < X.length; i++) {
            sum = (sum + X[i]);
        }

        mean = sum/X.length;

        for (int i = 0; i < X.length; i++) {
            sum2 = sum2 + Math.pow(X[i] - mean,2);
        }

        variance = sum2/ X.length;
    }

    public void createXScaledAndXNorm(){
        if ( variance!= 0) {
            // find min-max
            for (int i = 0; i < X.length; i++) {
                X_Scaled[i] = (X[i] - mean) / (Math.sqrt(variance + e));
                X_BatchNorm[i] = V * (X_Scaled[i]) + B;
            }
        }
    }

    public double[] normalise(double[] X){
        if( zScale)
        {
            if ( variance!= 0) {
                // find min-max
                for (int i = 0; i < X.length; i++) {
                    X[i] = (((X[i] - mean) / (Math.sqrt(variance)))) ;
                }
            }
        }
        else if( min_max) {
            double min = 0;
            double max = 0;
            for (int j = 0; j < X.length; j++) {
                if (X[j] < min)
                    min = X[j];
                if (X[j] > max)
                    max = X[j];
            }
            if (max != min) {
                // find min-max
                for (int i = 0; i < X.length; i++) {
                    X[i] = 1 * ((X[i] - min) / (max - min));
                }
            }
        }
        else {
            for (int i = 0; i < X.length; i++) {
                X[i] = 1 * ((X[i] ) / (Matrix.add(X)));
            }
        }
        return X;
    }

    @Override
    public double getOutput(){
        if( isResidual)
            return (Matrix.add(X) + y);
        else return y;
    }

    @Override
    public double getGradient(int batchIndex){
        return Gradient_Batch[batchIndex];
    }
}
