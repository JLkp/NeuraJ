package de.jlkp.ai.optimizer;

import de.jlkp.ai.Correction;
import de.jlkp.ai.cache.BackwardCache;
import de.jlkp.ai.cache.OptimizerCache;
import de.jlkp.ai.layer.DenseLayer;
import de.jlkp.ai.loss.LossFunction;
import de.jlkp.ai.utils.AiUtils;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.math3.linear.*;

import java.util.ArrayList;
import java.util.List;

@Slf4j
public class Adam implements Optimizer {
    LossFunction lossFunction;
    private double learningRate;
    private final double beta1 = 0.9;
    private final double beta2 = 0.999;
    private final double epsilon = 10e-8;

    private int t = 0; // time step, used for bias correction

    private List<BlockRealMatrix> Vdw;
    private List<RealVector> Vdb;

    private List<BlockRealMatrix> Sdw;
    private List<RealVector> Sdb;

    private List<DenseLayer> layers;

    public Adam(LossFunction lossFunction) {
        this.lossFunction = lossFunction;
    }

    /** Initializes the optimizer: creates lists for first and second momentum, builds corrections matrices for each layer*/
    public void initialize(List<DenseLayer> layers, double learningRate) {
        this.layers = layers;
        this.learningRate = learningRate;

        Vdw = new ArrayList<>();
        Vdb = new ArrayList<>();
        Sdw = new ArrayList<>();
        Sdb = new ArrayList<>();

        for (DenseLayer l : layers) {
            BlockRealMatrix weightMatrix = new BlockRealMatrix(l.getWeights().getData());
            RealVector biasVector = l.getBias();
            Vdw.add(new BlockRealMatrix(weightMatrix.getRowDimension(), weightMatrix.getColumnDimension()));
            Vdb.add(new ArrayRealVector(biasVector.getDimension()));

            Sdw.add(new BlockRealMatrix(weightMatrix.getRowDimension(), weightMatrix.getColumnDimension()));
            Sdb.add(new ArrayRealVector(biasVector.getDimension()));
        }
    }

    @Override
    public LossFunction getLossFunction() {
        return lossFunction;
    }

    /** Calculates the correction matrices for each layer of the network*/
    @Override
    public List<Correction> applyGradient(OptimizerCache optimizerCache) {
        t++;
        List<Correction> corrections = new ArrayList<>();  // updates for weights and biases

        RealMatrix outputVector = optimizerCache.getForwardCaches().getLast().getZ();
        outputVector = layers.getLast().getActivationFunction().activate(outputVector); // gets output of the last layer
        RealMatrix lossGradient = lossFunction.gradient(optimizerCache.getLabelVector(), outputVector); // calculates the first loss gradient

        int j = layers.size() - 1;
        // builds corrections for each layer, starting from the last one
        for (DenseLayer layer : layers.reversed()) {
            BackwardCache backwardCache = layer.backward(lossGradient, optimizerCache.getForwardCaches().removeLast());

            RealMatrix dw = backwardCache.getCorrection().getWeightsCorrection(); // get delta weights
            RealVector db = backwardCache.getCorrection().getBiasCorrection(); // get delta bias
            //log.info("Backward correction: {} \n {}", dw, db);

            Vdw.get(j).walkInOptimizedOrder(new DefaultRealMatrixChangingVisitor() {  // calculate first momentum for weights
                @Override
                public double visit(int row, int column, double value) {
                    return beta1 * value + (1 - beta1) * dw.getEntry(row, column);
                }
            });
            Vdb.get(j).walkInOptimizedOrder(new RealVectorChangingVisitor() {  // calculate first momentum for bias
                @Override
                public void start(int dimension, int start, int end) {

                }

                @Override
                public double visit(int index, double value) {
                    return beta1 * value + (1 - beta1) * db.getEntry(index);
                }

                @Override
                public double end() {
                    return 0;
                }
            });

            Sdw.get(j).walkInOptimizedOrder(new DefaultRealMatrixChangingVisitor() {  // calculate second momentum for weights
                @Override
                public double visit(int row, int column, double value) {
                    return beta2 * value + (1 - beta2) * dw.getEntry(row, column) * dw.getEntry(row, column);
                }
            });

            Sdb.get(j).walkInOptimizedOrder(new RealVectorChangingVisitor() { // calculate second momentum for bias
                @Override
                public void start(int dimension, int start, int end) {

                }

                @Override
                public double visit(int index, double value) {
                    return beta2 * value + (1 - beta2) * db.getEntry(index) * db.getEntry(index);
                }

                @Override
                public double end() {
                    return 0;
                }
            });

            // bias correction for weights and bias (because vdw and vdb, sdw and sdb are biased because initialization with 0)
            RealMatrix vdwCorrected = Vdw.get(j).scalarMultiply(1.0 / (1.0 - Math.pow(beta1, t)));  //TODO: check for possible optimizations
            RealVector vdbCorrected = Vdb.get(j).mapMultiply(1.0 / (1.0 - Math.pow(beta1, t))); //TODO: check for possible optimizations

            RealMatrix sdwCorrected = Sdw.get(j).scalarMultiply(1.0 / (1.0 - Math.pow(beta2, t))); //TODO: check for possible optimizations
            RealVector sdbCorrected = Sdb.get(j).mapMultiply(1.0 / (1.0 - Math.pow(beta2, t))); //TODO: check for possible optimizations

            // build correction matrices for weights and bias
            Correction correction = new Correction();
            correction.setWeightsCorrection(AiUtils.ebeDivide(vdwCorrected, AiUtils.ebePow(sdwCorrected, 0.5).scalarAdd(epsilon)).scalarMultiply(learningRate)); //TODO: check for possible optimizations
            correction.setBiasCorrection(vdbCorrected.ebeDivide(AiUtils.ebePow(sdbCorrected, 0.5).mapAdd(epsilon)).mapMultiply(learningRate)); // TODO: check for possible optimizations

            corrections.add(correction); // add one correction for each layer to list of corrections

            // set lossGradient for next layer
            lossGradient = backwardCache.getBackpropagationError();
            j--;


        }


        return corrections.reversed();
    }
}
