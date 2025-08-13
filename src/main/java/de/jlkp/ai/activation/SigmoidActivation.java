package de.jlkp.ai.activation;

import org.apache.commons.math3.linear.DefaultRealMatrixChangingVisitor;
import org.apache.commons.math3.linear.RealMatrix;

public class SigmoidActivation implements ActivationFunction {

    @Override
    public RealMatrix activate(RealMatrix input) {
        RealMatrix result = input.copy();
        result.walkInOptimizedOrder(new DefaultRealMatrixChangingVisitor() {
            @Override
            public double visit(int row, int column, double value) {
                return 1 / (1 + Math.exp(-value));
            }
        });
        return result;
    }

    @Override
    public RealMatrix derivative(RealMatrix Z) {
        RealMatrix sigmoid = Z.copy();
        sigmoid.walkInOptimizedOrder(new DefaultRealMatrixChangingVisitor() {
            @Override
            public double visit(int row, int column, double value) {
                double sig = 1.0 / (1.0 + Math.exp(-value)); // \sigma(z)
                return sig * (1.0 - sig); // \sigma(z) * (1 - \sigma(z))
            }
        });
        return sigmoid;
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
