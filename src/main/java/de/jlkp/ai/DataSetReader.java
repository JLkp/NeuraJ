package de.jlkp.ai;

import java.util.List;

public interface DataSetReader {
    double[][] getData();

    List<String> getLabels();
}
