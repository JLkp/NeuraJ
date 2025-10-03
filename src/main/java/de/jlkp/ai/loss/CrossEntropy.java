package de.jlkp.ai.loss;

import lombok.extern.slf4j.Slf4j;
import org.apache.commons.math3.linear.RealMatrix;

@Slf4j
public class CrossEntropy implements LossFunction {

    private static final double EPS = 1e-15;

    @Override
    public double loss(RealMatrix predicted, RealMatrix labels) {
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
        return sum / cols;
    }

    @Override
    public RealMatrix gradient(RealMatrix label, RealMatrix output) {
        return output.subtract(label).scalarMultiply(1.0 / output.getColumnDimension());
        //return output.subtract(label);
    }

}
