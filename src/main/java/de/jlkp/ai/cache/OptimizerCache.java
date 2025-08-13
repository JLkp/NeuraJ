package de.jlkp.ai.cache;

import lombok.Data;
import org.apache.commons.math3.linear.RealMatrix;

import java.util.List;

// wrapper class for the correction of each layer
@Data
public class OptimizerCache {
    List<ForwardCache> forwardCaches;  // all forward caches of the layers
    RealMatrix labelVector;

    public OptimizerCache(List<ForwardCache> forwardCaches, RealMatrix labelVector) {
        this.forwardCaches = forwardCaches;
        this.labelVector = labelVector;
    }
}
