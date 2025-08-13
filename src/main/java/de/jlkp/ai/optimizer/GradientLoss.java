package de.jlkp.ai.optimizer;

import de.jlkp.ai.Correction;
import de.jlkp.ai.cache.OptimizerCache;
import de.jlkp.ai.layer.DenseLayer;
import de.jlkp.ai.loss.LossFunction;
import lombok.extern.slf4j.Slf4j;

import java.util.ArrayList;
import java.util.List;

@Slf4j
public class GradientLoss implements Optimizer {
    LossFunction lossFunction; // The loss function to be used for the optimization
    double learningRate;

    public GradientLoss(LossFunction lossFunction, double learningRate) {
        this.lossFunction = lossFunction;
        this.learningRate = learningRate;
    }

    @Override
    public List<Correction> applyGradient(OptimizerCache optimizerCache) {
        List<Correction> corrections = new ArrayList<>();
        /*// log.info("{}", labelVector);
        RealMatrix outputVector = optimizerCache.getForwardCaches().getLast().getZ();
        outputVector = optimizerCache.getDenseLayers().getLast().getActivationFunction().activate(outputVector);
        RealMatrix lossGradient = lossFunction.gradient(optimizerCache.getLabelVector(), outputVector);


        for (DenseLayer layer : optimizerCache.getDenseLayers().reversed()) {
            BackwardCache backwardCache = layer.backward(lossGradient, optimizerCache.getForwardCaches().removeLast());
            lossGradient = backwardCache.getBackpropagationError();
            corrections.addFirst(backwardCache.getCorrection());
        }*/

        return corrections;
    }

    @Override
    public void initialize(List<DenseLayer> layers, double learningRate) {

    }

    @Override
    public LossFunction getLossFunction() {
        return null;
    }
}
