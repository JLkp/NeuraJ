package de.jlkp.ai.data;

import javax.xml.crypto.Data;
import java.util.List;

public class SimpleReader extends DataSet implements TrainingSet {
    private final String fileName;

    public SimpleReader(String fileName) {
        this.fileName = fileName;
    }

    public void readData() {
        // Implement reading data from a file



    }


    @Override
    public DataSet getData() {
        return null;
    }

    @Override
    public DataSet getData(List<String> labelEncoding) {
        return null;
    }

    @Override
    public List<String> getLabelNames() {
        return List.of();
    }
}
