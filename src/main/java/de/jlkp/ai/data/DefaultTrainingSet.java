package de.jlkp.ai.data;

import lombok.extern.slf4j.Slf4j;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

import java.util.List;

@Slf4j
public class DefaultTrainingSet implements TrainingSet {
    private RealMatrix data; // column by column, first column is label
    private List<String> labels; // all labels of the trainingset in order
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
        // log.info("{}", labelMatrix); TODO: comment in to test
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
