package de.jlkp.ai;

import de.jlkp.ai.layer.DenseLayer;
import de.jlkp.ai.optimizer.Optimizer;
import org.apache.commons.math3.linear.RealVector;

public interface NeuralNetwork {

    // lrFit: Wenn Wert da wird Learning Rate nach jeder Epoche um diesen Faktor multipliziert
    void train(TrainingSet data, int epochs, double learningRate, int batchSize, int miniBatchSize, TrainingSet validationSet, ReduceLROnPlateau reduceLROnPlateau);

    RealVector predict(RealVector input);

    double evaluate(TrainingSet data);

    void addHiddenLayer(DenseLayer layer);

    void compile(double maxInputValue, Optimizer optimizer); // TODO: optimizer und loss function hinzufügen
}
