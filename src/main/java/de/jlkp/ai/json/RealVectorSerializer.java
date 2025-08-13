package de.jlkp.ai.json;

import com.fasterxml.jackson.core.JsonGenerator;
import com.fasterxml.jackson.databind.JsonSerializer;
import com.fasterxml.jackson.databind.SerializerProvider;
import org.apache.commons.math3.linear.RealVector;

import java.io.IOException;

public class RealVectorSerializer extends JsonSerializer<RealVector> {
    @Override
    public void serialize(RealVector realVector, JsonGenerator jsonGenerator, SerializerProvider serializerProvider) throws IOException {
        jsonGenerator.writeObject(realVector.toArray());
    }
}
