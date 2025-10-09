package de.jlkp.ai.data;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/** Reads the CSV File, which is located at the given fileName*/
public class CsvDataSetReader implements DataSetReader {
    private final String fileName;

    public CsvDataSetReader(String fileName) {
        this.fileName = fileName;
    }

    /** Returns the features that are in the CSV file */
    @Override
    public double[][] getSamples() {
        List<double[]> sampleRows = new ArrayList<>(); // holds the content of each sample row

        // read the csv file, safes each line in a array that is a element of the sampleRows list
        try (BufferedReader br = new BufferedReader(new FileReader(fileName))) {
            br.readLine();
            String line;  // current line
            while ((line = br.readLine()) != null) {
                String[] parts = line.split(","); // csv separation with comma
                if (parts.length < 2) continue; // skip line, if there's no data

                double[] row = new double[parts.length - 1]; // create a new row for the samples
                for (int i = 1; i < parts.length; i++) {
                    row[i - 1] = Double.parseDouble(parts[i].trim()); // save each element of the row
                }
                sampleRows.add(row); // add the current sample row to the list of sample rows
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return sampleRows.toArray(new double[0][]);
    }

    /** Returns the labels to the samples from the CSV file*/
    @Override
    public List<String> getLabels() {
        List<String> labels = new ArrayList<>();

        try (BufferedReader br = new BufferedReader(new FileReader(fileName))) {
            br.readLine();
            String line;
            while ((line = br.readLine()) != null) {
                String[] parts = line.split(","); // csv separation with comma
                if (parts.length < 2) continue; // skip line, if there's no data

                labels.add(parts[0]); // save label at the front
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return labels;  // returns the list of labels
    }
}
