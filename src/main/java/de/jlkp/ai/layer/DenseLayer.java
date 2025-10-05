package de.jlkp.ai.layer;

import de.jlkp.ai.Correction;
import de.jlkp.ai.activation.ActivationFunction;
import de.jlkp.ai.cache.BackwardCache;
import de.jlkp.ai.cache.ForwardCache;
import de.jlkp.ai.utils.AiUtils;
import lombok.Getter;
import lombok.Setter;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.math3.linear.*;

import java.util.Random;

@Slf4j
public class DenseLayer {
    @Getter
    private final ActivationFunction activationFunction;  // activation function of this layer

    @Getter
    private final int neuronCount;

    @Getter
    @Setter
    private RealMatrix weights;

    @Getter
    @Setter
    private RealVector bias;


    public DenseLayer(ActivationFunction activationFunction, int neuronCount) {
        this.neuronCount = neuronCount;
        this.activationFunction = activationFunction;
    }

    /** Runs the input through the network and returns computed matrix */
    public RealMatrix forward(RealMatrix input, ForwardCache forwardCache) {
        if (input.getRowDimension() != weights.getColumnDimension()) {
            throw new IllegalArgumentException("Input row dimensions do not match count of neurons");
        }

        if (forwardCache != null) { // saves the input and z-values (after weights, before activation function) for backpropagation
            forwardCache.setInput(input.copy());
            forwardCache.setZ(AiUtils.addBias(weights.multiply(input), bias));
            return activationFunction.activate(forwardCache.getZ()).copy();
        }
        // act(input * w + b)
        return activationFunction.activate(AiUtils.addBias(weights.multiply(input), bias));
    }

    /** Builds the backwardCache for the layer*/
    public BackwardCache backward(RealMatrix backpropagationError, ForwardCache forwardCache) {
        BackwardCache backwardCache = new BackwardCache();
        Correction correction = new Correction();

        RealMatrix da = activationFunction.derivative(forwardCache.getZ()); // gradient of activation function and z-values
        RealMatrix grad = AiUtils.ebeMultiply(backpropagationError, da); // gradient of the cost function/loss function * da

        correction.setWeightsCorrection(grad.multiply(forwardCache.getInput().transpose())); // calculates the weight correction
        correction.setBiasCorrection(AiUtils.meanBias(grad)); // calculates the bias correction

        backwardCache.setBackpropagationError(weights.transpose().multiply(grad));
        backwardCache.setCorrection(correction);

        return backwardCache;
    }

    /** Initializes the layers weights and bias*/
    public void initialize(int preLayerNeuronCount) {
        this.weights = new Array2DRowRealMatrix(neuronCount, preLayerNeuronCount);

        Random random = new Random(42);  // TODO: HERE IS A SEED
        double wScale = activationFunction.getWeightInitScale(preLayerNeuronCount, neuronCount); // weight initialization depending on activation function (He/Xavier)
        double b = activationFunction.getbiasInit(); // bias initialization depending on activation function (usually 0)

        weights.walkInOptimizedOrder(new DefaultRealMatrixChangingVisitor() {

            @Override
            public double visit(int row, int column, double value) {
                return random.nextGaussian() * wScale; // creates normally distributed weights with mean 0 and stddev wScale
            }
        });

        bias = new ArrayRealVector(neuronCount, b); // creates bias vector with all values = b
    }

    /** Imports weights from json file into the layer*/
    public void importWeights(RealMatrix weights) {
        this.weights = weights;
    }

    /** Updates the layer weights and bias*/
    public void updateParameters(Correction correction) {
        if (correction.getWeightsCorrection() != null) {
            weights = weights.subtract(correction.getWeightsCorrection()); // weight = weight - learningRate * dE/dW
        }
        if (correction.getBiasCorrection() != null) {
            bias = bias.subtract(correction.getBiasCorrection()); // bias = bias - learningRate * dE/db
        }
    }
}
