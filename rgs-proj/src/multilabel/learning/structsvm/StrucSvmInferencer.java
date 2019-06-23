package multilabel.learning.structsvm;

import edu.illinois.cs.cogcomp.sl.core.AbstractInferenceSolver;
import edu.illinois.cs.cogcomp.sl.core.IInstance;
import edu.illinois.cs.cogcomp.sl.core.IStructure;
import edu.illinois.cs.cogcomp.sl.util.WeightVector;
import multilabel.evaluation.LossFunction;
import multilabel.learning.StructOutput;
import multilabel.learning.inferencer.SearchInferencer;

public class StrucSvmInferencer extends AbstractInferenceSolver {

	private static final long serialVersionUID = -889244124347219759L;
	
	SearchInferencer searchSolver;
	
	public StrucSvmInferencer(int bsize, int maxd) {
		searchSolver = new SearchInferencer(bsize, maxd);
	}
	

	@Override
	public IStructure getBestStructure(WeightVector weight, IInstance xi) throws Exception {
		StrucSvmInstance x = (StrucSvmInstance) xi;
		//StructOutput result = ilpSolver.inference(x.example, weight, null, false);
		StructOutput result = searchSolver.inference(x.example, weight, null, false);
		return result;
	}

	
	
	public double getTrainLoss(StructOutput pred, StructOutput truth) {
		//double loss = 1.0 - LossFunction.computeHammingAccuracy(pred, truth);
		//return loss;
		int[] crrWng = LossFunction.computeRightWrong(pred, truth);
		//int correct = crrWng[0];
		int wrong = crrWng[1];
		return wrong;
	}
	
	@Override
	public float getLoss(IInstance xi, IStructure ystari, IStructure yhati) {
		
		StrucSvmInstance x = (StrucSvmInstance) xi;
		StructOutput ystar = (StructOutput) ystari;
		StructOutput yhat = (StructOutput) yhati;
		
		double loss = getTrainLoss(yhat, ystar);
		return ((float)loss);
	}

	@Override
	public IStructure getLossAugmentedBestStructure(WeightVector weight,
			IInstance xi, IStructure ystari) throws Exception {
		
		StrucSvmInstance x = (StrucSvmInstance) xi;
		StructOutput ystar = (StructOutput) ystari;
		
		//StructOutput result = ilpSolver.inference(x.example, weight, ystar, true);
		StructOutput result = searchSolver.inference(x.example, weight, ystar, true);
		
		return result;
	}

}
