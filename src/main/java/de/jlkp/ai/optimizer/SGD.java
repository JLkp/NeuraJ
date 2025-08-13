package de.jlkp.ai.optimizer;

import de.jlkp.ai.Correction;
import de.jlkp.ai.ReduceLROnPlateau;
import de.jlkp.ai.activation.ActivationFunction;
import de.jlkp.ai.cache.OptimizerCache;
import de.jlkp.ai.layer.DenseLayer;
import de.jlkp.ai.loss.LossFunction;
import de.jlkp.ai.utils.AiUtils;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

import java.util.ArrayList;
import java.util.List;

@Slf4j
public class SGD implements Optimizer {
    LossFunction lossFunction; // loss function used for the optimization
    double learningRate; // learning rate for the optimization
    ReduceLROnPlateau reduceLROnPlateau; // learning rate scheduler for the optimization

    private List<RealMatrix> weights; // weights of the network
    private List<RealVector> biases; // biases of the network
    private List<ActivationFunction> activationFunctions; // activation functions of the network

    public SGD(LossFunction lossFunction, double learningRate, ReduceLROnPlateau reduceLROnPlateau) {
        this.lossFunction = lossFunction;
        this.learningRate = learningRate;
        this.reduceLROnPlateau = reduceLROnPlateau;
    }


    @Override
    public List<Correction> applyGradient(OptimizerCache optimizerCache) {
        //param forwardCaches: first forward cache is the input layer, last forward cache is the output layer
        // param labels: the labels of the current batch, the labels are the expected output of the network
        List<Correction> correction = new ArrayList<>();

        // TODO: Biases are currently not corrected in the SGD optimizer, so we return 0 as correction for the biases

        // calculate the correction matrices for the output layer
        RealMatrix aValues = activationFunctions.getLast().activate(optimizerCache.getForwardCaches().getLast().getZ()); // calculate a-values: act(z-values) with last forward cache
        RealMatrix lossDelta = lossFunction.gradient(aValues, optimizerCache.getLabelVector()); // calculate the error between the a-values and the input of the last forward cache
        RealMatrix delta = AiUtils.ebeMultiply(lossDelta, activationFunctions.getLast().derivative(optimizerCache.getForwardCaches().getLast().getZ())); // calculate the delta values: error * derivative of the activation function
        RealMatrix correctOutputWeights = delta.multiply(optimizerCache.getForwardCaches().getLast().getInput().transpose());

        int dimBiases = biases.getLast().getDimension();
        RealVector correctOutputBiases = new ArrayRealVector(dimBiases);
        correction.add(new Correction(correctOutputWeights, correctOutputBiases));


        // calculate the correction matrices for the hidden layers
        for (int i = optimizerCache.getForwardCaches().size() - 2; i >= 0; i--) {
/*
            log.info("Dim lossDelta: {}x{}, Dim delta: {}x{}, Dim correctOutputWeights: {}x{}, Input: {}x{}",
                    lossDelta.getRowDimension(), lossDelta.getColumnDimension(), delta.getRowDimension(), delta.getColumnDimension(),
                    correctOutputWeights.getRowDimension(), correctOutputWeights.getColumnDimension(),
                    forwardCaches.getLast().getInput().getRowDimension(), forwardCaches.getLast().getInput().getColumnDimension());
*/

            delta = AiUtils.ebeMultiply(weights.get(i + 1).transpose().multiply(delta),
                    activationFunctions.get(i).derivative(optimizerCache.getForwardCaches().get(i).getZ()));
            correctOutputWeights = delta.multiply(optimizerCache.getForwardCaches().get(i).getInput().transpose());
            dimBiases = biases.get(i).getDimension();
            correctOutputBiases = new ArrayRealVector(dimBiases);
            correction.add(new Correction(correctOutputWeights, correctOutputBiases));
        }

        // subtract correction matrices from the weights and biases
        for (int i = 0; i < correction.size(); i++) {
            RealMatrix weightCorrection = correction.get(i).getWeightsCorrection();
            RealVector biasCorrection = correction.get(i).getBiasCorrection();

            // update weights
            RealMatrix currentWeights = weights.get(weights.size() - 1 - i);
            currentWeights = currentWeights.subtract(weightCorrection.scalarMultiply(learningRate));  // update networks weights
            weights.set(weights.size() - 1 - i, currentWeights); // update own weights

            // update biases
            RealVector currentBiases = biases.get(biases.size() - 1 - i);
            currentBiases = currentBiases.subtract(biasCorrection.mapMultiply(learningRate));
            biases.set(biases.size() - 1 - i, currentBiases);
        }

        return correction;
    }

    @Override
    public void initialize(List<DenseLayer> layers, double learningRate) {

    }

    @Override
    public LossFunction getLossFunction() {
        return null;
    }

}
