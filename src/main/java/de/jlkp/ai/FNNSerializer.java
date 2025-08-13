package de.jlkp.ai;

import com.fasterxml.jackson.annotation.JsonAutoDetect;
import com.fasterxml.jackson.annotation.PropertyAccessor;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.module.SimpleModule;

import de.jlkp.ai.json.RealMatrixDeserializer;
import de.jlkp.ai.json.RealMatrixSerializer;
import de.jlkp.ai.json.RealVectorSerializer;

import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

import java.io.File;

public class FNNSerializer {
    private final ObjectMapper mapper;

    public FNNSerializer() {
        mapper = new ObjectMapper();
        mapper.setVisibility(PropertyAccessor.FIELD, JsonAutoDetect.Visibility.ANY);
        SimpleModule module = new SimpleModule();
        module.addSerializer(RealMatrix.class, new RealMatrixSerializer());
        module.addDeserializer(RealMatrix.class, new RealMatrixDeserializer());
        module.addSerializer(RealVector.class, new RealVectorSerializer());
        mapper.registerModule(module);
    }

    public String serializeToJson(FNN fnn) {
        try {
            return mapper.writeValueAsString(fnn);
        } catch (Exception e) {
            throw new RuntimeException("Failed to serialize FNN", e);
        }
    }

    public FNN deserializeFromJson(String json) {
        try {
            return mapper.readValue(json, FNN.class);
        } catch (Exception e) {
            throw new RuntimeException("Failed to deserialize FNN", e);
        }
    }

    public void saveToFile(FNN fnn, String filePath) {
        try {
            mapper.writerWithDefaultPrettyPrinter().writeValue(new File(filePath), fnn);
        } catch (Exception e) {
            throw new RuntimeException("Failed to save FNN to file", e);
        }
    }

    public FNN loadFromFile(String filePath) {
        try {
            return mapper.readValue(new File(filePath), FNN.class);
        } catch (Exception e) {
            throw new RuntimeException("Failed to load FNN from file", e);
        }
    }
}
