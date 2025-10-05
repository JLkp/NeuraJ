package de.jlkp.ai.utils;

import org.apache.commons.math3.linear.*;

import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;

public class AiUtils {
    /**
     * Erzeugt einen RealVector mit zufälligen Double-Werten im Intervall [min,max).
     *
     * @param dimension Dimension des Vektors
     * @param min       untere Grenze (inklusive)
     * @param max       obere Grenze (exklusive)
     * @return RealVector mit zufälligen Einträgen
     */
    public static RealVector randomRealVector(int dimension, double min, double max) {
        double[] data = new double[dimension];
        ThreadLocalRandom rnd = ThreadLocalRandom.current();
        for (int i = 0; i < dimension; i++) {
            data[i] = rnd.nextDouble(min, max);
        }
        return new ArrayRealVector(data, false);
    }

    public static RealMatrix randomRealMAtrix(int rows, int cols, double min, double max) {
        RealMatrix matrix = new Array2DRowRealMatrix(rows, cols);
        ThreadLocalRandom rnd = ThreadLocalRandom.current();
        matrix.walkInOptimizedOrder(new DefaultRealMatrixChangingVisitor() {
            @Override
            public double visit(int row, int column, double value) {
                return rnd.nextDouble(min, max);
            }
        });
        return matrix;
    }

    public static RealMatrix ebeMultiply(RealMatrix a, RealMatrix b) {
        if (a.getRowDimension() != b.getRowDimension() || a.getColumnDimension() != b.getColumnDimension()) {
            throw new IllegalArgumentException("Matrices must have the same dimensions for element-wise multiplication.");
        }

        RealMatrix result = a.copy();
        for (int i = 0; i < a.getRowDimension(); i++) {
            for (int j = 0; j < a.getColumnDimension(); j++) {
                result.setEntry(i, j, a.getEntry(i, j) * b.getEntry(i, j));
            }
        }
        return result;
    }

    public static RealMatrix ebeDivide(RealMatrix a, RealMatrix b) {
        if (a.getRowDimension() != b.getRowDimension() || a.getColumnDimension() != b.getColumnDimension()) {
            throw new IllegalArgumentException("Matrices must have the same dimensions for element-wise multiplication.");
        }

        RealMatrix result = a.copy();
        for (int i = 0; i < a.getRowDimension(); i++) {
            for (int j = 0; j < a.getColumnDimension(); j++) {
                result.setEntry(i, j, a.getEntry(i, j) / b.getEntry(i, j));
            }
        }
        return result;


    }

    public static BlockRealMatrix toBlockRealMatrix(RealMatrix a) {
        if(a instanceof BlockRealMatrix) {
            return (BlockRealMatrix)a;
        }
        return new BlockRealMatrix(a.getData());
    }

    public static BlockRealMatrix scalarMultiplyBlock(RealMatrix a, double scalar) {
        BlockRealMatrix result = toBlockRealMatrix(a);
        for (int i = 0; i < a.getRowDimension(); i++) {
            for (int j = 0; j < a.getColumnDimension(); j++) {
                result.setEntry(i, j, a.getEntry(i, j) * scalar);
            }
        }
        return result;
    }

    public static BlockRealMatrix ebeMultiplyBlock(RealMatrix a, RealMatrix b) {
        return toBlockRealMatrix(ebeMultiply(a, b));
    }

    public static BlockRealMatrix ebeDivideBlock(RealMatrix a, RealMatrix b) {
        return toBlockRealMatrix(ebeDivide(a, b));
    }

    public static double max(RealMatrix matrix) {
        double max = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < matrix.getRowDimension(); i++) {
            for (int j = 0; j < matrix.getColumnDimension(); j++) {
                max = Math.max(max, matrix.getEntry(i, j));
            }
        }
        return max;
    }

    public static RealMatrix addBias(RealMatrix w, RealVector bias) {
        RealMatrix result = w.copy();
        for (int i = 0; i < result.getColumnDimension(); i++) {
            result.setColumnVector(i, result.getColumnVector(i).add(bias));
        }
        return result;
    }

    public static RealVector meanBias(RealMatrix b) {
        if (b.getColumnDimension() == 0) {
            throw new IllegalArgumentException("Matrix must have at least one column.");
        }
        int colDim = b.getColumnDimension();
        RealVector meanBias = new ArrayRealVector(b.getRowDimension());
        for (int i = 0; i < b.getRowDimension(); i++) {
            double sum = 0;
            for (int j = 0; j < colDim; j++) {
                sum += b.getEntry(i, j);
            }
            meanBias.setEntry(i, sum / colDim);
        }
        return meanBias;
    }

    /**
     * Mischt die Spalten von zwei Matrizen (inputs und labels) synchron und in-place.
     * Die Zuordnung zwischen inputs und labels bleibt erhalten.
     *
     * @param inputs RealMatrix mit Eingabedaten (z. B. 784 x 25000)
     * @param labels RealMatrix mit Labels (z. B. 5 x 25000)
     * @throws IllegalArgumentException wenn Matrizen null sind oder unterschiedliche Spaltenanzahlen haben
     */
    public static void shuffle(RealMatrix inputs, RealMatrix labels) {
        if (inputs == null || labels == null) {
            throw new IllegalArgumentException("Inputs und Labels dürfen nicht null sein");
        }
        if (inputs.getColumnDimension() != labels.getColumnDimension()) {
            throw new IllegalArgumentException("Inputs und Labels müssen die gleiche Anzahl an Spalten haben");
        }

        int cols = inputs.getColumnDimension();
        Random rand = new Random(42); // TODO: HERE IS A SEED

        // Fisher-Yates-Shuffle
        for (int i = cols - 1; i > 0; i--) {
            int j = rand.nextInt(i + 1);
            // Vertausche Spalten i und j in inputs und labels
            swapColumns(inputs, i, j);
            swapColumns(labels, i, j);
        }
    }

    public static RealMatrix ebePow(RealMatrix matrix, double exponent) {
        RealMatrix result = matrix.copy();
        for (int i = 0; i < result.getRowDimension(); i++) {
            for (int j = 0; j < result.getColumnDimension(); j++) {
                result.setEntry(i, j, Math.pow(result.getEntry(i, j), exponent));
            }
        }
        return result;
    }

    public static RealVector ebePow(RealVector vector, double exponent) {
        RealVector result = vector.copy();
        for (int i = 0; i < result.getDimension(); i++) {
            result.setEntry(i, Math.pow(result.getEntry(i), exponent));
        }
        return result;
    }


    /**
     * Hilfsmethode zum Vertauschen von zwei Spalten in einer Matrix in-place.
     *
     * @param matrix Die zu verändernde Matrix
     * @param col1   Index der ersten Spalte
     * @param col2   Index der zweiten Spalte
     */
    private static void swapColumns(RealMatrix matrix, int col1, int col2) {
        for (int row = 0; row < matrix.getRowDimension(); row++) {
            double temp = matrix.getEntry(row, col1);
            matrix.setEntry(row, col1, matrix.getEntry(row, col2));
            matrix.setEntry(row, col2, temp);
        }
    }
}
