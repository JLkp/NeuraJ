package de.jlkp.ai.data;

import java.util.List;

public interface DataSetReader {
    double[][] getSamples();

    List<String> getLabels();
}
