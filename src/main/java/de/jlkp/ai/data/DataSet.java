package de.jlkp.ai.data;

import lombok.Data;
import org.apache.commons.math3.linear.RealMatrix;

@Data
public class DataSet {
    RealMatrix samples;
    RealMatrix labels;
}
