package de.jlkp.ai.json;

import com.fasterxml.jackson.core.JsonGenerator;
import com.fasterxml.jackson.databind.JsonSerializer;
import com.fasterxml.jackson.databind.SerializerProvider;
import org.apache.commons.math3.linear.RealMatrix;

import java.io.IOException;

public class RealMatrixSerializer extends JsonSerializer<RealMatrix> {
    @Override
    public void serialize(RealMatrix realMatrix, JsonGenerator jsonGenerator, SerializerProvider serializerProvider) throws IOException {
        jsonGenerator.writeObject(realMatrix.getData());
    }
}
