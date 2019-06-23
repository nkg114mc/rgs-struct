package multilabel.learning.cost;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.ObjectInputStream;

import multilabel.instance.Example;
import multilabel.instance.Featurizer;
import multilabel.instance.OldWeightVector;
import multilabel.learning.RegressionCostFuncLearning;
import multilabel.learning.search.OldSearchState;
import weka.classifiers.functions.LinearRegression;
import weka.classifiers.functions.SimpleLinearRegression;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

public class RegressionCostFunction extends CostFunction {
/*
	private Featurizer featurizer;
	private LibSVM model;
	
	public RegressionCostFunction(Featurizer fizer) {
		featurizer = fizer;
	}
	
	public void loadModel(String path) {
		File modelf = new File(path);
		if (!modelf.exists()) {
			throw new RuntimeException("Model " + path + " does not exist!");
		}
		
		//  public Object LoadModel(String file){
        try{
            ObjectInputStream ois = new ObjectInputStream(new FileInputStream(path));
            Object classifier = (ois.readObject());
            ois.close();
            model = (LibSVM) classifier;//new J48();
            System.out.println("Loaded weka classifier: " + model.toString());
            System.out.println("Weka model file: " + path);
        } catch(IOException e){
            e.printStackTrace();
        } catch(ClassNotFoundException e){
            e.printStackTrace();
        }
	}
	
	public double getCost(OldSearchState state, Example ex) {
		OldWeightVector fv = featurizer.getFeatureVector(ex, state.getOutput());
		double sc = predict(fv);
		return sc;
	}
	
	public Instances getInstances(OldWeightVector fv) {
		FastVector attrs = RegressionCostFuncLearning.getAttributes(fv.getMaxLength());
		Instances wekaDataSet = new Instances("regression_learning", attrs, 1);
		wekaDataSet.setClass((Attribute)(attrs.lastElement()));
		return wekaDataSet;
	}
	
	public double predict(OldWeightVector featureVector) {
		int i;
		double predicted = 0;
		Instance instance = new Instance(featureVector.getMaxLength());

		for (i = 0; i < featureVector.getMaxLength(); i++) {
			String name = new String("feature" + (i+1));
			//instance.setValue(inst.attribute(i), featureVector.get(i)));
			instance.setValue(i, featureVector.get(i)); 
			//.setValue(inst.attribute(i), featureVector.get(i)));
		}
		
		//System.out.println("instantce dem = " + inst.numAttributes());
		Instances inst = getInstances(featureVector);
		instance.setDataset(inst);
		
		// do predicting
		try {
			if (model == null) {
				throw new RuntimeException("Model null!!!!");
			}
			predicted = model.classifyInstance(instance);
			return predicted;
		} catch (Exception e) {
			e.printStackTrace();
		}
		return predicted;
	}
	
	
	//////////////////
	
	public void train() {
		//LibSVM libsvmTrainer = ;
	}
*/
}
