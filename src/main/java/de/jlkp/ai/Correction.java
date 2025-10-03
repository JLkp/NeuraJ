package de.jlkp.ai;

import lombok.Data;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

@Data
public class Correction {
    RealMatrix weightsCorrection; // correction of the weights: the updated weights after applying the gradient
    RealVector biasCorrection; // correction of the biases: the updated biases after applying the gradient

    public Correction(RealMatrix weightsCorrection, RealVector biasCorrection) {
        this.weightsCorrection = weightsCorrection;
        this.biasCorrection = biasCorrection;
    }

    public Correction() {}
}
