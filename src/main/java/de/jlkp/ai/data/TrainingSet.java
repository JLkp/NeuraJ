package de.jlkp.ai.data;

import java.util.List;

/** interface that defines the methods a dataset needs*/
public interface TrainingSet {
    DataSet getData();

    DataSet getData(List<String> labelEncoding);

    List<String> getLabelNames();
}
