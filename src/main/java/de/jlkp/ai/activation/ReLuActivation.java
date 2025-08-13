package de.jlkp.ai.activation;

import org.apache.commons.math3.linear.DefaultRealMatrixChangingVisitor;
import org.apache.commons.math3.linear.RealMatrix;

public class ReLuActivation implements ActivationFunction {
    @Override
    public RealMatrix activate(RealMatrix input) {
        RealMatrix result = input.copy();
        result.walkInOptimizedOrder(new DefaultRealMatrixChangingVisitor() {
            @Override
            public double visit(int row, int column, double value) {
                return Math.max(0, value);
            }
        });
        return result;
    }

    @Override
    public RealMatrix derivative(RealMatrix Z) {
        RealMatrix relu = Z.copy();
        relu.walkInOptimizedOrder(new DefaultRealMatrixChangingVisitor() {
            @Override
            public double visit(int row, int column, double value) {
                return value > 0 ? 1.0 : 0.0;
            }
        });
        return relu;

    }

    @Override
    public double getWeightInitScale(int inputSize, int outputSize) {
        return Math.sqrt(2.0 / inputSize);
    }

    @Override
    public double getbiasInit() {
        return 0.01;
    }
}
