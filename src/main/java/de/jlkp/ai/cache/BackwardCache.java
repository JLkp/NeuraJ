package de.jlkp.ai.cache;

import de.jlkp.ai.Correction;
import lombok.Data;
import org.apache.commons.math3.linear.RealMatrix;

@Data
public class BackwardCache {
    Correction correction;
    RealMatrix backpropagationError;
}
