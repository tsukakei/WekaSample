package classifier;

import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.functions.LinearRegression;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;

/**
 * Created by TK on 6/22/17.
 */
public class WekaRegression {
    protected LinearRegression regModel;
    protected Instances instances;

    public static void main(String[] args) {
        WekaRegression regression = new WekaRegression();
        regression.training();
        regression.regression();
    }

    public void training() {
        try {
            System.out.println("Start training");
            // Generate a regression model
            ConverterUtils.DataSource source = new ConverterUtils.DataSource("housing.arff");
            instances = source.getDataSet();
            instances.setClassIndex(instances.numAttributes() - 1);
            regModel = new LinearRegression();
            regModel.buildClassifier(instances);
            // Evaluate the generated classifier
            Evaluation eval = new Evaluation(instances);
            eval.evaluateModel(regModel, instances);
            System.out.println(eval.toSummaryString());
            System.out.println("Finish training");
        } catch (Exception e) {
            e.printStackTrace();
        }

    }

    public void regression() {
        try {
            Attribute crim = new Attribute("crim", 0);
            Attribute zn = new Attribute("zn", 1);
            Attribute indus = new Attribute("indus", 2);
            Attribute chas = new Attribute("chas", 3);
            Attribute nox = new Attribute("nox", 4);
            Attribute rm = new Attribute("rm", 5);
            Attribute age = new Attribute("age", 6);
            Attribute dis = new Attribute("dis", 7);
            Attribute rad = new Attribute("rad", 8);
            Attribute tax = new Attribute("tax", 9);
            Attribute ptratio = new Attribute("ptratio", 10);
            Attribute b = new Attribute("b", 11);
            Attribute lstat = new Attribute("lstat", 12);
            Attribute medv = new Attribute("mdev", 13);
            Instance instance = new DenseInstance(instances.numAttributes());
            instance.setValue(crim, 0.00632);
            instance.setValue(zn, 18);
            instance.setValue(indus, 2.31);
            instance.setValue(chas, 0);
            instance.setValue(nox, 0.538);
            instance.setValue(rm, 6.5750);
            instance.setValue(age, 65.2);
            instance.setValue(dis, 4.09);
            instance.setValue(rad, 1);
            instance.setValue(tax, 296);
            instance.setValue(ptratio, 15.3);
            instance.setValue(b, 396.9);
            instance.setValue(lstat, 4.98);
            instance.setDataset(instances);
            double result = regModel.classifyInstance(instance);
            System.out.println(result);
            System.out.println("Finish classifying");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
