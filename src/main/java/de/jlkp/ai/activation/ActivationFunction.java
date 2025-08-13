package de.jlkp.ai.activation;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.annotation.JsonSubTypes;
import com.fasterxml.jackson.annotation.JsonTypeInfo;
import org.apache.commons.math3.linear.RealMatrix;

@JsonTypeInfo(
        use = JsonTypeInfo.Id.NAME,
        include = JsonTypeInfo.As.PROPERTY,
        property = "type")
@JsonSubTypes({
        @JsonSubTypes.Type(value = ReLuActivation.class, name = "relu"),
        @JsonSubTypes.Type(value = SigmoidActivation.class, name = "sigmoid"),
        @JsonSubTypes.Type(value = SoftmaxActivation.class, name = "softmax")
        // Weitere Implementierungen hier einfügen
})
public interface ActivationFunction {
    RealMatrix activate(RealMatrix input);

    RealMatrix derivative(RealMatrix label);

    @JsonIgnore
    double getWeightInitScale(int inputSize, int outputSize);

    @JsonIgnore
    double getbiasInit();
}
