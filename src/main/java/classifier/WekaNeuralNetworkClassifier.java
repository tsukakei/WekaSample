package classifier;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.SMO;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;

/**
 * Created by TK on 6/22/17.
 */
public class WekaNeuralNetworkClassifier {
    protected Classifier classifier;
    protected Instances instances;
    protected Instances train;
    protected Instances test;

    public static void main(String[] args) {
        WekaNeuralNetworkClassifier weka = new WekaNeuralNetworkClassifier();
        weka.classify();
    }

    public void classify() {
        try {
            System.out.println("Start training");
            // Generate a neural network classifier
            ConverterUtils.DataSource source = new ConverterUtils.DataSource("iris.arff");
            instances = source.getDataSet();
            instances.setClassIndex(instances.numAttributes() - 1);
            int trainSize = (int) Math.round(instances.numInstances() * 0.9);
            int testSize = instances.numInstances() - trainSize;
            train = new Instances(instances, 0, trainSize);
            test = new Instances(instances, trainSize, testSize);
            //Instance of NN
            MultilayerPerceptron mlp = new MultilayerPerceptron();
            //Setting Parameters
            mlp.setLearningRate(0.1);
            mlp.setMomentum(0.2);
            mlp.setTrainingTime(2000);
            mlp.setHiddenLayers("3");
            mlp.buildClassifier(train);
            Evaluation eval = new Evaluation(train);
            eval.evaluateModel(mlp, train);
            System.out.println(eval.toSummaryString());
            System.out.println("Finish training");
            
            // Evaluate the generated classifier
            System.out.println("Start evaluating");
            eval = new Evaluation(test);
            eval.evaluateModel(mlp, test);
            System.out.println(eval.toSummaryString());
            System.out.println("Finish evaluating");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
