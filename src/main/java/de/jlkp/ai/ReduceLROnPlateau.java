package de.jlkp.ai;

import lombok.extern.slf4j.Slf4j;

import java.util.ArrayList;
import java.util.List;

@Slf4j
public class ReduceLROnPlateau {
    private final int patience;
    private final double factor;
    private final double threshold;
    private List<Double>  valLossHistory;

    public ReduceLROnPlateau(int patience, double factor, double threshold) {
        this.patience = patience;
        this.factor = factor;
        this.threshold = threshold;
        this.valLossHistory = new ArrayList<>();
    }

    public void trackValLoss(double valLoss) {
        if( valLossHistory.size() >= patience) {
            valLossHistory.removeFirst(); // remove the oldest loss if we exceed patience
            valLossHistory.add(valLoss); // add the new validation loss
        }else{
            valLossHistory.add(valLoss); // add the new validation loss
        }
    }

    public double updateLr(double lr){
        log.info("LR updated");
        // wenn sich der ValLoss der letzten patience Epochen nicht um threshhold verbessert hat, wird die lr um factor reduziert
        if (valLossHistory.size() < patience) {
            return lr; // not enough history to decide
        }

        double sumLoss = 0.0;
        for(Double l : valLossHistory) {
            sumLoss += l;
        }

        if(sumLoss / patience > threshold) {
            double newLr = lr * factor;
            valLossHistory.clear(); // reset history after reducing learning rate
            return newLr;
        }

        return lr; // no change to learning rate if condition is not met
    }



}
