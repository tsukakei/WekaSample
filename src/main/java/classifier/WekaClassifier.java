package classifier;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.SMO;
import weka.core.*;
import weka.core.converters.ConverterUtils;
import weka.filters.unsupervised.instance.RemovePercentage;

/**
 * Created by TK on 6/22/17.
 */
public class WekaClassifier {
    protected Classifier classifier;
    protected Instances instances;
    protected Instances train;
    protected Instances test;

    public static void main(String[] args) {
        WekaClassifier weka = new WekaClassifier();
        weka.training();
//        weka.classify();
    }

    protected void training() {
         try {
             System.out.println("Start training");
             // Generate a classifier
             ConverterUtils.DataSource source = new ConverterUtils.DataSource("iris.arff");
             instances = source.getDataSet();
             instances.setClassIndex(instances.numAttributes() - 1);
             int trainSize = (int) Math.round(instances.numInstances() * 0.9);
             int testSize = instances.numInstances() - trainSize;
             train = new Instances(instances, 0, trainSize);
             test = new Instances(instances, trainSize, testSize);
             classifier = new SMO();
             classifier.buildClassifier(train);
             // Evaluate the generated classifier
             Evaluation eval = new Evaluation(train);
             eval.evaluateModel(classifier, train);
             System.out.println(eval.toSummaryString());
             System.out.println("Finish training");

             System.out.println("Start evaluating");
             eval = new Evaluation(test);
             eval.evaluateModel(classifier, test);
             System.out.println(eval.toSummaryString());
             System.out.println("Finish evaluating");
         } catch (Exception e) {
             e.printStackTrace();
         }
    }

    protected void classify() {
        try {
            System.out.println("Start classifying");
            System.out.println("IrisSetosa(0.0)?: " + classifier.classifyInstance(getIrisSetosa()));
            System.out.println("IrisVersicolor(1.0)?: " + classifier.classifyInstance(getIrisVersicolor()));
            System.out.println("IrisVirginica(2.0)?: " + classifier.classifyInstance(getIrisVirginica()));
            System.out.println("Finish classifying");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private Instance getIrisData(double sepal_l, double sepal_w, double petal_l, double petal_w) {
        Attribute sepal_length = new Attribute("sepal_length", 0);
        Attribute sepal_width = new Attribute("sepal_width", 1);
        Attribute petal_length = new Attribute("petal_length", 2);
        Attribute petal_width = new Attribute("petal_width", 3);
        Instance instance = new DenseInstance(instances.numAttributes());
        instance.setValue(sepal_length, sepal_l);
        instance.setValue(sepal_width, sepal_w);
        instance.setValue(petal_length, petal_l);
        instance.setValue(petal_width, petal_w);
        instance.setDataset(instances);
        return instance;
    }

    private Instance getIrisSetosa() {
        return getIrisData(5.1, 3.5, 1.4, 0.2);
    }

    private Instance getIrisVersicolor() {
        return getIrisData(7.0, 3.2, 4.7, 1.4);
    }

    private Instance getIrisVirginica() {
        return getIrisData(6.3, 3.3, 6.0, 2.5);
    }
}
