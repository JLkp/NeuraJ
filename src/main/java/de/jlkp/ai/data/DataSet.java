package de.jlkp.ai.data;

import lombok.Data;
import org.apache.commons.math3.linear.RealMatrix;

@Data
public class DataSet {
    RealMatrix samples;  // features that get feed into network
    RealMatrix labels;  // targets the network fits to
}
