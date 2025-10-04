package de.jlkp.ai.data;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class CsvDataSetReader implements DataSetReader {
    private final String fileName;

    public CsvDataSetReader(String fileName) {
        this.fileName = fileName;
    }

    @Override
    public double[][] getData() {
        List<String> labels = new ArrayList<>();
        List<double[]> dataRows = new ArrayList<>();

        try (BufferedReader br = new BufferedReader(new FileReader(fileName))) {
            br.readLine();
            String line;
            while ((line = br.readLine()) != null) {
                String[] parts = line.split(","); // csv separation with comma
                if (parts.length < 2) continue; // skip line, if no data

                labels.add(parts[0]); // safe label at front

                double[] row = new double[parts.length - 1];
                for (int i = 1; i < parts.length; i++) {
                    row[i - 1] = Double.parseDouble(parts[i].trim());
                }
                dataRows.add(row);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return dataRows.toArray(new double[0][]);
    }

    @Override
    public List<String> getLabels() {
        List<String> labels = new ArrayList<>();
        List<double[]> dataRows = new ArrayList<>();

        try (BufferedReader br = new BufferedReader(new FileReader(fileName))) {
            br.readLine();
            String line;
            while ((line = br.readLine()) != null) {
                String[] parts = line.split(","); // csv separation with comma
                if (parts.length < 2) continue; // skip line, if no data

                labels.add(parts[0]); // save label at the front

                double[] row = new double[parts.length - 1];
                for (int i = 1; i < parts.length; i++) {
                    row[i - 1] = Double.parseDouble(parts[i].trim());
                }
                dataRows.add(row);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return labels;
    }
}
