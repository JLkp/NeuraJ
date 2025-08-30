package de.jlkp.ai;

import com.fasterxml.jackson.annotation.JsonProperty;
import de.jlkp.ai.cache.ForwardCache;
import de.jlkp.ai.cache.OptimizerCache;
import de.jlkp.ai.data.DataSet;
import de.jlkp.ai.data.DefaultTrainingSet;
import de.jlkp.ai.data.TrainingSet;
import de.jlkp.ai.layer.DenseLayer;
import de.jlkp.ai.optimizer.Optimizer;
import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

import static de.jlkp.ai.utils.AiUtils.shuffle;

@Slf4j
public class FNN implements NeuralNetwork, Serializable {
    @Getter
    @JsonProperty("InputDimension")
    private final int inputDim;


    @JsonProperty("Layer")
    private List<DenseLayer> layer;

    private double maxInputValue;

    @JsonProperty("Optimizer")
    private Optimizer optimizer;

    private List<String> labelNames;


    public FNN(int inputDim) {
        layer = new ArrayList<>();
        this.inputDim = inputDim;
    }

    @Override
    public void train(DefaultTrainingSet data, int epochs, double learningRate, int batchSize, TrainingSet validationSet) {
        validationSet = null; // Not used in this implementation
        optimizer.initialize(layer, learningRate);
        DataSet ds = data.getData();
        labelNames = data.getLabelNames();
        RealMatrix samples = ds.getSamples().scalarMultiply(1.0 / maxInputValue);
        log.info("{}x{}", samples.getRowDimension(), samples.getColumnDimension());
        RealMatrix labels = ds.getLabels();

        // log.info("{}", ds);

        for (int i = 0; i < epochs; i++) {
            shuffle(samples, labels);
            int batches = samples.getColumnDimension() / batchSize;
            double loss = 0.0;

            for (int batchIndex = 0; batchIndex < batches; batchIndex++) {
                RealMatrix inputBatch = samples.getSubMatrix(0, samples.getRowDimension() - 1, batchIndex * batchSize, (batchIndex + 1) * batchSize - 1);
                RealMatrix labelBatch = labels.getSubMatrix(0, labels.getRowDimension() - 1, batchIndex * batchSize, (batchIndex + 1) * batchSize - 1);

                List<ForwardCache> forwardCaches = new ArrayList<>();
                for (DenseLayer layer : layer) {
                    ForwardCache forwardCache = new ForwardCache();
                    inputBatch = layer.forward(inputBatch, forwardCache);
                    forwardCaches.add(forwardCache);
                }

                //log.info("Fehler: {}", lossFunction.loss(labelBatch, inputBatch));
                loss += optimizer.getLossFunction().loss(labelBatch, inputBatch);

                OptimizerCache optimizerCache = new OptimizerCache(forwardCaches, labelBatch);
                List<Correction> corrections = optimizer.applyGradient(optimizerCache);

                int j = 0;
                for (DenseLayer l : layer) {
                    //log.info(formatMatrix(corrections.get(j).getWeightsCorrection(), 4));
                    l.updateParameters(corrections.get(j));
                    j++;
                }

            }
            log.info("Loss: {}", loss / batches);

        }
    }

    @Deprecated
    public void train(DefaultTrainingSet trainingSet, int epochs, double learningRate, TrainingSet validationSet) {
        log.info("Training started... ");

        DataSet ds = trainingSet.getData();
        RealMatrix samples = ds.getSamples().scalarMultiply(1 / maxInputValue);
        RealMatrix labels = ds.getLabels();

        optimizer.initialize(layer, learningRate);

        for (int i = 0; i < epochs; i++) {
            for (int sampleIndex = 0; sampleIndex < samples.getColumnDimension(); sampleIndex++) {
                RealVector sample = samples.getColumnVector(sampleIndex);
                RealVector label = labels.getColumnVector(sampleIndex);

                RealMatrix outputMatrix = MatrixUtils.createColumnRealMatrix(sample.toArray());
                RealMatrix labelMatrix = MatrixUtils.createColumnRealMatrix(label.toArray());

                // Forward pass
                List<ForwardCache> forwardCaches = new ArrayList<>();

                for (DenseLayer layer : layer) {
                    ForwardCache forwardCache = new ForwardCache();
                    outputMatrix = layer.forward(outputMatrix, forwardCache);
                    forwardCaches.add(forwardCache);

                }
                //log.info("Loss: {}", optimizer.getLossFunction().loss(labelMatrix, outputMatrix));
                OptimizerCache optimizerCache = new OptimizerCache(forwardCaches, labelMatrix);

                // Backward pass
                List<Correction> correction = optimizer.applyGradient(optimizerCache);
                //log.info("Corr: {}", correction.size());

                // update weights and biases
                int j = 0;
                for (DenseLayer l : layer) {
                    l.updateParameters(correction.get(j));
                    j++;
                }


            }
        }


    }

    private RealMatrix forward(RealMatrix data) {
        RealMatrix outputVector = data.copy();
        for (DenseLayer layer : layer) {
            outputVector = layer.forward(outputVector, null);
        }
        return outputVector;
    }

    @Override
    public RealVector predict(RealVector input) {
        if (input.getDimension() != inputDim) {
            throw new IllegalArgumentException("Input dimension mismatch");
        }
        RealMatrix outputVector = MatrixUtils.createColumnRealMatrix(input.toArray());

        // Normalize the input vector
        outputVector = outputVector.scalarMultiply(1.0 / maxInputValue);

        for (DenseLayer layer : layer) {
            outputVector = layer.forward(outputVector, null);
        }

        return outputVector.getColumnVector(0);

    }

    @Override
    public double evaluate(TrainingSet data) {

        int correctCount = 0;
        RealMatrix predictions = forward(data.getData().getSamples().scalarMultiply(1.0 / maxInputValue));
        RealMatrix labels = data.getData(labelNames).getLabels();
        int[] corr = new int[labels.getRowDimension()];
        for (int i = 0; i < predictions.getColumnDimension(); i++) {
            RealVector prediction = predictions.getColumnVector(i);
            RealVector label = labels.getColumnVector(i);

            // Check if the predicted label matches the actual label
            if (prediction.getMaxIndex() == label.getMaxIndex()) {
                correctCount++;
                corr[prediction.getMaxIndex()] += 1;
            }
        }
        log.info("Evaluated correct count: {}", corr);
        return (double) correctCount / predictions.getColumnDimension();
    }

    @Override
    public void addHiddenLayer(DenseLayer layer) {
        this.layer.add(layer);
    }


    @Override
    public void compile(double maxInputValue, Optimizer optimizer) {
        this.maxInputValue = maxInputValue;
        int preNeurons = inputDim;
        for (DenseLayer layer : layer) {
            layer.initialize(preNeurons);
            preNeurons = layer.getNeuronCount();
        }
        if (optimizer != null) {
            this.optimizer = optimizer;
        } else {
            throw new IllegalArgumentException("Optimizer must not be null");
        }

    }

    public String summary() {
        StringBuilder sb = new StringBuilder();
        sb.append("\n");
        sb.append("------------ FNN Summary ------------\n");
        sb.append("Input Dimension: ").append(inputDim).append("\n");
        sb.append("Layers:\n");
        for (DenseLayer layer : layer) {
            sb.append(layer).append("\n");
        }
        sb.append("-------------------------------------");
        return sb.toString();
    }

}
