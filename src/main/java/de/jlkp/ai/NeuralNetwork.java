package de.jlkp.ai;

import de.jlkp.ai.data.DefaultTrainingSet;
import de.jlkp.ai.data.TrainingSet;
import de.jlkp.ai.layer.DenseLayer;
import de.jlkp.ai.optimizer.Optimizer;
import org.apache.commons.math3.linear.RealVector;

public interface NeuralNetwork {

    // lrFit: Wenn Wert da wird Learning Rate nach jeder Epoche um diesen Faktor multipliziert
    void train(DefaultTrainingSet data, int epochs, double learningRate, int batchSize, TrainingSet validationSet);

    RealVector predict(RealVector input);

    double evaluate(TrainingSet data);

    void addHiddenLayer(DenseLayer layer);

    void compile(double maxInputValue, Optimizer optimizer);
}
