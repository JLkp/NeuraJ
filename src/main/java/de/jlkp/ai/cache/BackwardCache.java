package de.jlkp.ai.cache;

import de.jlkp.ai.Correction;
import lombok.Data;
import org.apache.commons.math3.linear.RealMatrix;

@Data
public class BackwardCache {
    Correction correction; // Cache keeps deltas for weights and biases TODO: change name of variable, is misleading
    RealMatrix backpropagationError; // error that the layer propagates to the next layer
}
