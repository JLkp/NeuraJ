package de.jlkp.ai.data;

import java.util.List;


public interface TrainingSet {
    DataSet getData();

    DataSet getData(List<String> labelEncoding);

    List<String> getLabelNames();
}
