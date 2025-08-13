package de.jlkp.ai.activation;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;

public class TanhActivation implements ActivationFunction {
    @Override
    public RealMatrix activate(RealMatrix input) {
        double[][] in = input.getData();
        int rows = in.length, cols = in[0].length;
        double[][] out = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                out[i][j] = Math.tanh(in[i][j]);
            }
        }
        return new Array2DRowRealMatrix(out);
    }

    @Override
    public RealMatrix derivative(RealMatrix label) {
        double[][] lab = label.getData();
        int rows = lab.length, cols = lab[0].length;
        double[][] out = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                out[i][j] = 1.0 - (lab[i][j] * lab[i][j]);
            }
        }
        return new Array2DRowRealMatrix(out);
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
