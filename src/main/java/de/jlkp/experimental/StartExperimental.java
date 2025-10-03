package de.jlkp.experimental;

import de.jlkp.ai.FNN;
import de.jlkp.ai.activation.ReLuActivation;
import de.jlkp.ai.activation.SoftmaxActivation;
import de.jlkp.ai.data.CsvDataSetReader;
import de.jlkp.ai.data.DataSetReader;
import de.jlkp.ai.data.DefaultTrainingSet;
import de.jlkp.ai.layer.DenseLayer;
import de.jlkp.ai.loss.CrossEntropy;
import de.jlkp.ai.optimizer.Adam;
import lombok.extern.slf4j.Slf4j;

@Slf4j
public class StartExperimental {

    public static void main(String[] args) {
        log.info("Starting experimental code...");


        long start = System.nanoTime();
        currentBuild();
//        smallTest();
        long end = System.nanoTime();
        long duration = (end - start);  //divide by 1000000 to get milliseconds.
        log.info("Duration: {} ms", duration / 1_000_000);


    }

    // this is how the current neural network is build, trained and used
    public static void currentBuild(){
        // import of trainset
        String trainPath = StartExperimental.class
                .getClassLoader()
                .getResource("pictures_train.csv")
                .getPath();
        DefaultTrainingSet t = new DefaultTrainingSet();
        DataSetReader dsr = new CsvDataSetReader(trainPath);
        t.importData(dsr);

        // import of testset
        String testPath = StartExperimental.class
                .getClassLoader()
                .getResource("pictures_test.csv")
                .getPath();
        DefaultTrainingSet t2 = new DefaultTrainingSet();
        DataSetReader dsr2 = new CsvDataSetReader(testPath);
        t2.importData(dsr2);


        FNN model = new FNN(784);
        model.addHiddenLayer(new DenseLayer(new ReLuActivation(), 128));
        model.addHiddenLayer(new DenseLayer(new ReLuActivation(), 64));
        model.addHiddenLayer(new DenseLayer(new SoftmaxActivation(), 5));

        model.compile(255, new Adam(new CrossEntropy()));
        model.train(t, 5, 0.01, 10, true);

        log.info("{}", model.evaluate(t2, false));

    }

    public static void smallTest(){
        // import of trainset
        String trainPath = StartExperimental.class
                .getClassLoader()
                .getResource("iris_train.csv")
                .getPath();
        DefaultTrainingSet t = new DefaultTrainingSet();
        DataSetReader dsr = new CsvDataSetReader(trainPath);
        t.importData(dsr);

        // import of testset
        String testPath = StartExperimental.class
                .getClassLoader()
                .getResource("iris_test.csv")
                .getPath();
        DefaultTrainingSet t2 = new DefaultTrainingSet();
        DataSetReader dsr2 = new CsvDataSetReader(testPath);
        t2.importData(dsr2);


        FNN model = new FNN(4);
        model.addHiddenLayer(new DenseLayer(new ReLuActivation(), 2));
        model.addHiddenLayer(new DenseLayer(new SoftmaxActivation(), 3));

        model.compile(255, new Adam(new CrossEntropy()));
        model.train(t, 2, 0.01, 10, true);

        log.info("{}", model.evaluate(t2, false));

    }

}
