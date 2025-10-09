package de.jlkp.ai.loss;

import lombok.extern.slf4j.Slf4j;
import org.apache.commons.math3.linear.RealMatrix;

@Slf4j
public class CategoricalCrossEntropy implements LossFunction {

    private static final double EPS = 1e-15;

    /** Computes the loss of the network*/
    @Override
    public double loss(RealMatrix predicted, RealMatrix labels) {
        if (labels.getRowDimension() != predicted.getRowDimension()) {
            throw new IllegalArgumentException("Matrices have to have same row dimension.");
        }else if(labels.getColumnDimension() != predicted.getColumnDimension()) {
            throw new IllegalArgumentException("Matrices have to have same column dimension.");
        }

        int rows = predicted.getRowDimension();
        int cols = predicted.getColumnDimension();
        double sum = 0;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                double y = labels.getEntry(i, j);
                double p = predicted.getEntry(i, j);
                sum -= y * Math.log(p + EPS);
            }
        }
        return sum / cols;  // returns sum(y * log(p)) / N
    }

    /** Computes the gradient of the loss of the network*/
    @Override
    public RealMatrix gradient(RealMatrix predicted, RealMatrix labels) {
        return predicted.subtract(labels).scalarMultiply(1.0 / predicted.getColumnDimension());
    }

}
