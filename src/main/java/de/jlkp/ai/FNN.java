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

    /** Method that trains the network*/
    @Override
    public void train(DefaultTrainingSet data, int epochs, double learningRate, int batchSize, boolean verbose) {
        optimizer.initialize(layer, learningRate); // Initialize the optimizer with the layers and learning rate to fit the networks needs
        DataSet ds = data.getData(); // Get the dataset from the training set
        labelNames = data.getLabelNames();
        RealMatrix samples = ds.getSamples().scalarMultiply(1.0 / maxInputValue); // normalized the input data
//        log.info("{}x{}", samples.getRowDimension(), samples.getColumnDimension());
        RealMatrix labels = ds.getLabels();

        // loop for training epochs
        for (int i = 0; i < epochs; i++) {
            shuffle(samples, labels);
            int batches = samples.getColumnDimension() / batchSize;  // get number of batches
            double loss = 0.0;

            // loop for each batch
            for (int batchIndex = 0; batchIndex < batches; batchIndex++) {
                RealMatrix inputBatch = samples.getSubMatrix(0, samples.getRowDimension() - 1, batchIndex * batchSize, (batchIndex + 1) * batchSize - 1);
                RealMatrix labelBatch = labels.getSubMatrix(0, labels.getRowDimension() - 1, batchIndex * batchSize, (batchIndex + 1) * batchSize - 1);

                // Forward pass through all layers, storing the forward caches for backpropagation
                List<ForwardCache> forwardCaches = new ArrayList<>();
                for (DenseLayer layer : layer) {
                    ForwardCache forwardCache = new ForwardCache();
                    inputBatch = layer.forward(inputBatch, forwardCache);
                    forwardCaches.add(forwardCache);
                }

                loss += optimizer.getLossFunction().loss(labelBatch, inputBatch); // accumulate the loss for this batch

                // Create optimizer cache and apply gradients to get corrections for each layer
                OptimizerCache optimizerCache = new OptimizerCache(forwardCaches, labelBatch);
                List<Correction> corrections = optimizer.applyGradient(optimizerCache);

                // Update each layer's parameters using the calculated corrections
                int j = 0;
                for (DenseLayer l : layer) {
                    l.updateParameters(corrections.get(j));
                    j++;
                }

            }
            if(verbose){ // print loss if verbose is true
                log.info("Loss: {}", loss / batches);
            }

        }
    }

    /** forwards the input through the network and returns the vector from the last layer*/
    private RealMatrix forward(RealMatrix data) {
        RealMatrix outputVector = data.copy();
        for (DenseLayer layer : layer) {
            outputVector = layer.forward(outputVector, null);
        }
        return outputVector;
    }

    /** Predicts the input*/
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

    /** Evaluates the accuracy of the network*/
    @Override
    public double evaluate(TrainingSet data, boolean verbose) {

        int correctCount = 0;
        RealMatrix predictions = forward(data.getData().getSamples().scalarMultiply(1.0 / maxInputValue)); // run normalized data through the network
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
        if(verbose){
            log.info("Evaluated correct count: {}", corr);
        }
        return (double) correctCount / predictions.getColumnDimension();
    }

    /** Adds a hidden layer to the network*/
    @Override
    public void addHiddenLayer(DenseLayer layer) {
        this.layer.add(layer);
    }

    /** Compiles the network, initializing weights and biases, setting the optimizer*/
    @Override
    public void compile(double maxInputValue, Optimizer optimizer) {
        this.maxInputValue = maxInputValue;

        // creates weight matrices and bias vectors for each layer
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

    /** Prints summary of the network*/
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
