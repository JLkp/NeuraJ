package de.jlkp.ai.optimizer;

import de.jlkp.ai.Correction;
import de.jlkp.ai.cache.BackwardCache;
import de.jlkp.ai.cache.OptimizerCache;
import de.jlkp.ai.layer.DenseLayer;
import de.jlkp.ai.loss.LossFunction;
import de.jlkp.ai.utils.AiUtils;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

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

    private List<RealMatrix> Vdw;
    private List<RealVector> Vdb;

    private List<RealMatrix> Sdw;
    private List<RealVector> Sdb;

    private List<DenseLayer> layers;

    public Adam(LossFunction lossFunction) {
        this.lossFunction = lossFunction;
    }

    public void initialize(List<DenseLayer> layers, double learningRate) {
        this.layers = layers;
        this.learningRate = learningRate;

        Vdw = new ArrayList<>();
        Vdb = new ArrayList<>();
        Sdw = new ArrayList<>();
        Sdb = new ArrayList<>();

        for (DenseLayer l : layers) {
            RealMatrix weightMatrix = l.getWeights();
            RealVector biasVector = l.getBias();
            Vdw.add(new Array2DRowRealMatrix(weightMatrix.getRowDimension(), weightMatrix.getColumnDimension()));
            Vdb.add(new ArrayRealVector(biasVector.getDimension()));

            Sdw.add(new Array2DRowRealMatrix(weightMatrix.getRowDimension(), weightMatrix.getColumnDimension()));
            Sdb.add(new ArrayRealVector(biasVector.getDimension()));
        }
    }

    @Override
    public LossFunction getLossFunction() {
        return lossFunction;
    }

    @Override
    public List<Correction> applyGradient(OptimizerCache optimizerCache) {
        t++;
        List<Correction> corrections = new ArrayList<>();  // updates for weights and biases
        // log.info("{}", labelVector);
        RealMatrix outputVector = optimizerCache.getForwardCaches().getLast().getZ();
        outputVector = layers.getLast().getActivationFunction().activate(outputVector);
        //log.info("Output vector: {}", outputVector);
        //log.info("Output vector: {}", optimizerCache.getLabelVector());
        RealMatrix lossGradient = lossFunction.gradient(optimizerCache.getLabelVector(), outputVector);

        int j = layers.size() - 1;
        for (DenseLayer layer : layers.reversed()) {
            BackwardCache backwardCache = layer.backward(lossGradient, optimizerCache.getForwardCaches().removeLast());

            RealMatrix dw = backwardCache.getCorrection().getWeightsCorrection();
            RealVector db = backwardCache.getCorrection().getBiasCorrection();
            //log.info("Backward correction: {} \n {}", dw, db);


            Vdw.set(j, Vdw.get(j).scalarMultiply(beta1).add(dw.scalarMultiply(1 - beta1)));
            Vdb.set(j, Vdb.get(j).mapMultiply(beta1).add(db.mapMultiply(1 - beta1)));

            Sdw.set(j, Sdw.get(j).scalarMultiply(beta2).add(AiUtils.ebeMultiply(dw, dw).scalarMultiply(1 - beta2)));
            Sdb.set(j, Sdb.get(j).mapMultiply(beta2).add(db.ebeMultiply(db).mapMultiply(1 - beta2)));


            RealMatrix vdwCorrected = Vdw.get(j).scalarMultiply(1.0 / (1.0 - Math.pow(beta1, t)));
            RealVector vdbCorrected = Vdb.get(j).mapMultiply(1.0 / (1.0 - Math.pow(beta1, t)));

            RealMatrix sdwCorrected = Sdw.get(j).scalarMultiply(1.0 / (1.0 - Math.pow(beta2, t)));
            RealVector sdbCorrected = Sdb.get(j).mapMultiply(1.0 / (1.0 - Math.pow(beta2, t)));


            Correction correction = new Correction();
            correction.setWeightsCorrection(AiUtils.ebeDivide(vdwCorrected, AiUtils.ebePow(sdwCorrected, 0.5).scalarAdd(epsilon)).scalarMultiply(learningRate));
            correction.setBiasCorrection(vdbCorrected.ebeDivide(AiUtils.ebePow(sdbCorrected, 0.5).mapAdd(epsilon)).mapMultiply(learningRate));

            corrections.add(correction);

            // set lossGradient for next layer
            lossGradient = backwardCache.getBackpropagationError();
            j--;


        }


        return corrections.reversed();
    }
}
