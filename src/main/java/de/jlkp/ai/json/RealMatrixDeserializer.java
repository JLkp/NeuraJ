package de.jlkp.ai.json;

import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.databind.DeserializationContext;
import com.fasterxml.jackson.databind.JsonDeserializer;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;

import java.io.IOException;

public class RealMatrixDeserializer extends JsonDeserializer<RealMatrix> {

    @Override
    public RealMatrix deserialize(JsonParser jsonParser,
                                  DeserializationContext deserializationContext) throws IOException {
        double[][] data = jsonParser.readValueAs(double[][].class);
        return MatrixUtils.createRealMatrix(data);
    }
}
