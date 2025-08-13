package de.jlkp.ai.loss;

import lombok.extern.slf4j.Slf4j;
import org.apache.commons.math3.linear.RealMatrix;

@Slf4j
public class CrossEntropyVector implements LossFunction { // ermittelt den Cross Entropy Loss für Matrix der Form [n, 1] (ein Vektor)

    private static final double EPSILON = 1e-15; // small value to avoid log(0)

    @Override
    public double loss(RealMatrix label, RealMatrix output) {
        if (label.getColumnDimension() != output.getColumnDimension()) {
            throw new IllegalArgumentException("Label and output must have the same number of columns.");
        }

        double sum = 0.0;
        for (int i = 0; i < label.getRowDimension(); i++) {
            double y = label.getEntry(i, 0);
            double p = Math.max(Math.min(output.getEntry(0, i), 1.0 - EPSILON), EPSILON);
            sum += y * Math.log(p);

        }
        return -sum;
    }

    @Override
    public RealMatrix gradient(RealMatrix label, RealMatrix output) {
        if (label.getColumnDimension() != output.getColumnDimension()) {
            throw new IllegalArgumentException("Label and output must have the same number of columns.");
        }

        return output.subtract(label);
    }
}
