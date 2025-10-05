package de.jlkp.ai.loss;

import lombok.extern.slf4j.Slf4j;
import org.apache.commons.math3.linear.DefaultRealMatrixChangingVisitor;
import org.apache.commons.math3.linear.RealMatrix;

@Slf4j
public class MSE implements LossFunction {
    @Override
    public double loss(RealMatrix label, RealMatrix output) {
        if (label.getRowDimension() != output.getRowDimension()) {
            throw new IllegalArgumentException("Matrices have to have same row dimension.");
        }else if(label.getColumnDimension() != output.getColumnDimension()) {
            throw new IllegalArgumentException("Matrices have to have same column dimension.");
        }

        return label.walkInOptimizedOrder(new DefaultRealMatrixChangingVisitor() {
            private double sum = 0;

            @Override
            public double visit(int row, int column, double value) {
                double diff = value - output.getEntry(row, column);
                sum += diff * diff;
                return value;
            }

            @Override
            public double end() {
                return (sum / (label.getRowDimension() * label.getColumnDimension())); // return mean squared error
            }
        });
    }

    @Override
    public RealMatrix gradient(RealMatrix label, RealMatrix output) {
        int B = label.getColumnDimension(); // Batchgröße
        int outputDim = label.getRowDimension();
        //log.info("{}x{}", label.getRowDimension(), label.getColumnDimension());
        return output.subtract(label).scalarMultiply(2.0 / (B * outputDim)); // -2/(B*outputDim) * (Y - \hat{Y})
    }
}
