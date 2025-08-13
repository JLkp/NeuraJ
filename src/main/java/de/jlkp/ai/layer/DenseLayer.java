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
    private final ActivationFunction activationFunction;

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

    public RealMatrix forward(RealMatrix input, ForwardCache forwardCache) {
        if (input.getRowDimension() != weights.getColumnDimension()) {
            throw new IllegalArgumentException("Input row dimensions do not match count of neurons");
        }

        if (forwardCache != null) {
            forwardCache.setInput(input.copy());
            forwardCache.setZ(AiUtils.addBias(weights.multiply(input), bias));
            return activationFunction.activate(forwardCache.getZ()).copy();
        }
        // act(input * w + b)
        return activationFunction.activate(AiUtils.addBias(weights.multiply(input), bias));
    }

    public BackwardCache backward(RealMatrix backpropagationError, ForwardCache forwardCache) {
        BackwardCache backwardCache = new BackwardCache();
        Correction correction = new Correction();

        RealMatrix da = activationFunction.derivative(forwardCache.getZ());
        RealMatrix grad = AiUtils.ebeMultiply(backpropagationError, da);

        // log.info("Gradient: {}", grad.scalarMultiply(0.1));


        correction.setWeightsCorrection(grad.multiply(forwardCache.getInput().transpose()));
        correction.setBiasCorrection(AiUtils.meanBias(grad));

        backwardCache.setBackpropagationError(weights.transpose().multiply(grad));
        backwardCache.setCorrection(correction);

        return backwardCache;
    }

    public void initialize(int preLayerNeuronCount) {
        this.weights = new Array2DRowRealMatrix(neuronCount, preLayerNeuronCount);

        Random random = new Random();
        double wScale = activationFunction.getWeightInitScale(preLayerNeuronCount, neuronCount);
        double b = activationFunction.getbiasInit();

        weights.walkInOptimizedOrder(new DefaultRealMatrixChangingVisitor() {

            @Override
            public double visit(int row, int column, double value) {
                return random.nextGaussian() * wScale;
            }
        });

        bias = new ArrayRealVector(neuronCount, b);
    }

    public void importWeights(RealMatrix weights) {
        this.weights = weights;
    }

    public void updateParameters(Correction correction) {
        if (correction.getWeightsCorrection() != null) {
            weights = weights.subtract(correction.getWeightsCorrection());
        }
        if (correction.getBiasCorrection() != null) {
            bias = bias.subtract(correction.getBiasCorrection());
        }
    }
}
