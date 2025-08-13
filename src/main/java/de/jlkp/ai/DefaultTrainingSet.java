package de.jlkp.ai;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

import java.util.List;

public class DefaultTrainingSet implements TrainingSet {
    private RealMatrix data; // Spaltenweise, oberste Zeile ist Label
    private List<String> labels;
    private List<String> uniqueLabels;

    public void importData(DataSetReader reader) {
        data = MatrixUtils.createRealMatrix(reader.getData());
        data = data.transpose();
        labels = reader.getLabels();
    }

    public DataSet getData() {
        DataSet ds = new DataSet();
        ds.setSamples(data);

        uniqueLabels = labels.stream().distinct().toList();
        RealMatrix labelMatrix = MatrixUtils.createRealMatrix(uniqueLabels.size(), data.getColumnDimension());

        int index = 0;
        for (String label : labels) {
            RealVector hotVector;
            int labelIndex = uniqueLabels.indexOf(label);
            hotVector = MatrixUtils.createRealVector(new double[uniqueLabels.size()]);
            hotVector.setEntry(labelIndex, 1.0);

            labelMatrix.setColumnVector(index, hotVector);
            index++;
        }
        ds.setLabels(labelMatrix);

        return ds;
    }

    public DataSet getData(List<String> labelEncoding) {
        DataSet ds = new DataSet();
        ds.setSamples(data);

        RealMatrix labelMatrix = MatrixUtils.createRealMatrix(labelEncoding.size(), data.getColumnDimension());

        int index = 0;
        for (String label : labels) {
            RealVector hotVector;
            int labelIndex = labelEncoding.indexOf(label);
            hotVector = MatrixUtils.createRealVector(new double[labelEncoding.size()]);
            hotVector.setEntry(labelIndex, 1.0);

            labelMatrix.setColumnVector(index, hotVector);
            index++;
        }
        ds.setLabels(labelMatrix);

        return ds;
    }

    public List<String> getLabelNames() {
        return uniqueLabels;
    }
}
