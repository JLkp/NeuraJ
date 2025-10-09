package de.jlkp.ai.data;

import lombok.extern.slf4j.Slf4j;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

import java.util.List;

/** the default dataset which is used through the whole project as the format for data. */
@Slf4j
public class DefaultTrainingSet implements TrainingSet {
    private RealMatrix samples; // column by column, each column is one sample
    private List<String> labels; // all labels of the trainingset in order
    private List<String> uniqueLabels;  // unique labels of the trainingset

    /** imports data through reader of CSV file*/
    public void importData(DataSetReader reader) {
        samples = MatrixUtils.createRealMatrix(reader.getSamples());
        samples = samples.transpose();  // transpose to have each row (before transposition column) as one sample
        labels = reader.getLabels();
    }

    /** returns the samples and corresponding labels (one-hot-encoded) in a dataset object*/
    public DataSet getData() {
        DataSet ds = new DataSet();  // dataset with samples and one-hot-encoded labels
        ds.setSamples(samples);

        uniqueLabels = labels.stream().distinct().toList();
        RealMatrix labelMatrix = MatrixUtils.createRealMatrix(uniqueLabels.size(), samples.getColumnDimension());

        int index = 0;
        for (String label : labels) {
            RealVector hotVector;  // current one-hot-encoded vector
            int labelIndex = uniqueLabels.indexOf(label); // get the index which should be hot
            hotVector = MatrixUtils.createRealVector(new double[uniqueLabels.size()]); // new vector
            hotVector.setEntry(labelIndex, 1.0); // set the correct index to hot (1.0)

            labelMatrix.setColumnVector(index, hotVector); // add the one-hot-encoded vector to the label matrix
            index++;
        }
        ds.setLabels(labelMatrix); // set the label matrix in the dataset

        return ds;
    }

    /** returns the samples and corresponding labels (one-hot-encoded) in a dataset object*/
    public DataSet getData(List<String> labelEncoding) {
        DataSet ds = new DataSet();
        ds.setSamples(samples);

        RealMatrix labelMatrix = MatrixUtils.createRealMatrix(labelEncoding.size(), samples.getColumnDimension());

        int index = 0;
        for (String label : labels) {
            RealVector hotVector; // current one-hot-encoded vector
            int labelIndex = labelEncoding.indexOf(label); // get the index which should be hot
            hotVector = MatrixUtils.createRealVector(new double[labelEncoding.size()]); // new vector
            hotVector.setEntry(labelIndex, 1.0); // set the correct index to hot (1.0)

            labelMatrix.setColumnVector(index, hotVector);  // add the one-hot-encoded vector to the label matrix
            index++;
        }
        ds.setLabels(labelMatrix); // set the label matrix in the dataset

        return ds;
    }

    /** returns the unique labels of the dataset*/
    public List<String> getLabelNames() {
        return uniqueLabels;
    }


}
