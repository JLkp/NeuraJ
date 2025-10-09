package de.jlkp.ai.loss;

import lombok.extern.slf4j.Slf4j;
import org.apache.commons.math3.linear.DefaultRealMatrixChangingVisitor;
import org.apache.commons.math3.linear.RealMatrix;

/**
 * Mean Squared Error Loss Function
 */
@Slf4j
public class MSE implements LossFunction {

    @Override
    public double loss(RealMatrix predicted, RealMatrix labels) {
        if (labels.getRowDimension() != predicted.getRowDimension()) {
            throw new IllegalArgumentException("Matrices have to have same row dimension.");
        }else if(labels.getColumnDimension() != predicted.getColumnDimension()) {
            throw new IllegalArgumentException("Matrices have to have same column dimension.");
        }

        return labels.walkInOptimizedOrder(new DefaultRealMatrixChangingVisitor() {
            private double sum = 0;

            @Override
            public double visit(int row, int column, double value) {
                double diff = value - predicted.getEntry(row, column);
                sum += diff * diff;
                return value;
            }

            @Override
            public double end() {
                return (sum / (labels.getRowDimension() * labels.getColumnDimension())); // return mean squared error
            }
        });
    }

    @Override
    public RealMatrix gradient(RealMatrix predicted, RealMatrix labels) {
        int B = labels.getColumnDimension(); // Batchgröße
        int outputDim = labels.getRowDimension();
        //log.info("{}x{}", labels.getRowDimension(), labels.getColumnDimension());
        return predicted.subtract(labels).scalarMultiply(2.0 / (B * outputDim)); // -2/(B*outputDim) * (Y - \hat{Y})
    }
}
