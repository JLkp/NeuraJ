package de.jlkp.ai;

import de.jlkp.ai.data.DefaultTrainingSet;
import de.jlkp.ai.data.TrainingSet;
import de.jlkp.ai.layer.DenseLayer;
import de.jlkp.ai.optimizer.Optimizer;
import org.apache.commons.math3.linear.RealVector;

public interface NeuralNetwork {

    /** Method that trains the network*/
    void train(DefaultTrainingSet data, int epochs, double learningRate, int batchSize, boolean verbose);

    /** Predict the input*/
    RealVector predict(RealVector input);

    /** Evaluates the accuracy of the network*/
    double evaluate(TrainingSet data, boolean verbose);

    /** Adds a hidden layer to the network*/
    void addHiddenLayer(DenseLayer layer);

    /** Compiles the network, initializing weights and biases, setting the optimizer*/
    void compile(double maxInputValue, Optimizer optimizer);
}
