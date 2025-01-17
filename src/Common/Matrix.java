package src.Common;

public class Matrix {
    public static double dot(double[] A, double[] B){
        if( A.length != B.length) {
            System.out.println("Error ==>> dot product incorrect size" );
        }
        double dotProduct = 0;
        for(int i=0; i<A.length; i++){
            dotProduct = dotProduct + (A[i] * B[i]);
        }
        return dotProduct;
    }

    public static double add(double[] A){
        double sum = 0;
        for(int i=0; i<A.length; i++){
            sum = sum + (A[i]);
        }
        return sum;
    }

    public static double[] scalar(double[] A, double scalar){
        double []A_Scaled = new double[A.length];

        for(int i=0; i<A.length; i++){
            A_Scaled[i] = (A[i] * scalar);
        }
        return A_Scaled;
    }

    public static double[] deepCopy(double[] A){
        double []A_Copy = new double[A.length];
        for(int i=0; i< A.length; i++) {
            System.arraycopy(A, 0, A_Copy, 0, A.length);
        }
        return A_Copy;
    }
}
