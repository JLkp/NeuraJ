package de.jlkp.ai.activation;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;

public class SoftmaxActivation implements ActivationFunction {

    //private RealMatrix lastOutput;

    @Override
    public RealMatrix activate(RealMatrix input) {
        int rows = input.getRowDimension();
        int cols = input.getColumnDimension();
        RealMatrix result = MatrixUtils.createRealMatrix(rows, cols);

        for (int j = 0; j < cols; j++) {
            // 1. Maximalwert ermitteln
            double max = Double.NEGATIVE_INFINITY;
            for (int i = 0; i < rows; i++) {
                max = Math.max(max, input.getEntry(i, j));
            }
            // 2. Exponentialwerte und Summe berechnen
            double sum = 0;
            double[] exps = new double[rows];
            for (int i = 0; i < rows; i++) {
                double e = Math.exp(input.getEntry(i, j) - max);
                exps[i] = e;
                sum += e;
            }
            // 3. Normieren
            for (int i = 0; i < rows; i++) {
                result.setEntry(i, j, exps[i] / sum);
            }
        }

        //this.lastOutput = result;
        return result;
    }

    @Override
    public RealMatrix derivative(RealMatrix label) {
        return MatrixUtils.createRealMatrix(label.getRowDimension(), label.getColumnDimension()).scalarMultiply(0).scalarAdd(1.0);
    }

    @Override
    public double getWeightInitScale(int inputSize, int outputSize) {
        return Math.sqrt(2.0 / (inputSize + outputSize));
    }

    @Override
    public double getbiasInit() {
        return 0;
    }
}
