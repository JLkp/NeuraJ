package de.jlkp.ai.optimizer;

import de.jlkp.ai.Correction;
import de.jlkp.ai.cache.OptimizerCache;
import de.jlkp.ai.layer.DenseLayer;
import de.jlkp.ai.loss.LossFunction;

import java.util.List;

public interface Optimizer {
    // Calculates the correction matrices
    List<Correction> applyGradient(final OptimizerCache optimizerCache);

    void initialize(List<DenseLayer> layers, double learningRate);  // initializes the optimizer with the layers and learning rate

    LossFunction getLossFunction();

}
