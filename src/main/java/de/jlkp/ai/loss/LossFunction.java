package de.jlkp.ai.loss;

import com.fasterxml.jackson.annotation.JsonSubTypes;
import com.fasterxml.jackson.annotation.JsonTypeInfo;
import org.apache.commons.math3.linear.RealMatrix;

@JsonTypeInfo(
        use = JsonTypeInfo.Id.NAME,
        include = JsonTypeInfo.As.PROPERTY,
        property = "type")
@JsonSubTypes({
        @JsonSubTypes.Type(value = MSE.class, name = "mse")
})
public interface LossFunction {
    /** Computes the loss of the network*/
    double loss(RealMatrix label, RealMatrix output);

    /** Computes the gradient of the loss of the network*/
    RealMatrix gradient(RealMatrix label, RealMatrix output);
}
