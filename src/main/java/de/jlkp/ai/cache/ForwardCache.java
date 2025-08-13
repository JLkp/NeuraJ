package de.jlkp.ai.cache;

import lombok.Data;
import org.apache.commons.math3.linear.RealMatrix;

@Data
public class ForwardCache {
    private RealMatrix input; // input of the layer: the output of the previous layer or the input data
    private RealMatrix z; //z-values of the layer: after the linear transformation but before the activation function
}
