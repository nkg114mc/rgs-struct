package essvm;

import java.util.HashMap;

import edu.illinois.cs.cogcomp.sl.core.IInstance;
import edu.illinois.cs.cogcomp.sl.core.IStructure;
import edu.illinois.cs.cogcomp.sl.core.SLProblem;
import edu.illinois.cs.cogcomp.sl.util.IFeatureVector;
import edu.illinois.cs.cogcomp.sl.util.WeightVector;


public class GurobiQpSolverWrapper {
/*
	//// init at first
	GRBEnv env = null;
	
	//// only prepared once for each dataset
	GRBModel  model = null;
	GRBVar[] wvars = null;
	GRBVar[] svars = null;
	GRBQuadExpr obj = null;
	
	HashMap<IInstance, GRBVar> exampleSlackMap;
	
	int optCallCnt = 0;
	
	
	public GurobiQpSolverWrapper() {
		try {
			env = new GRBEnv("qp.log");
		} catch (GRBException e) {
			System.out.println("Error code: " + e.getErrorCode() + ". " + e.getMessage());
		}
	}
	
	public void prepareAtFirst(SLProblem problem, WeightVector w) {
		
		try {
			model = new GRBModel(env);

			// Create variables
			wvars = new GRBVar[w.getLength()];
			for (int i = 0; i < wvars.length; i++) {
				String wname = "w_" + String.valueOf(i);
				wvars[i] = model.addVar(0.0, 1.0, 0.0, GRB.CONTINUOUS, wname);
			}
			
			exampleSlackMap = new HashMap<IInstance, GRBVar>();
			svars = new GRBVar[problem.size()];
			for (int j = 0; j < svars.length; j++) {
				String sname = "s_" + String.valueOf(j);
				svars[j] = model.addVar(0.0, 1.0, 0.0, GRB.CONTINUOUS, sname);
				exampleSlackMap.put(problem.instanceList.get(j), svars[j]);
			}

			//// 2. construct objective
			double svmC = 0.01;
			obj = new GRBQuadExpr();
			// 2-norm
			for (int i = 0; i < wvars.length; i++) {
				obj.addTerm(0.5, wvars[i], wvars[i]); // 2-norm square
			}
			// slack variables
			for (int j = 0; j < problem.size(); j++) {
				obj.addTerm(svmC, svars[j], svars[j]); // slack variable (L2 reguirization)
			}

			model.setObjective(obj);

		} catch (GRBException e) {
			e.printStackTrace();
		}
	}
	
	public void doSolveQP(SLProblem problem, SSVMWorkSet increamentalWorkset, WeightVector w) {
		
		try {
			
			optCallCnt++;
			
			//// 1. prepare
			if (optCallCnt == 1) {
				prepareAtFirst(problem, w);
			}

			//// 3. adding constraints
			for (int j = 0; j < problem.size(); j++) {
				
				IInstance x = problem.instanceList.get(j);
				GRBVar xslack = exampleSlackMap.get(x);
				assert(xslack != null);
				
				if (increamentalWorkset.existInstance(x)) {
					
					HashMap<IStructure, SSVMConstraint> instWorkSet = increamentalWorkset.cachedConstrs.get(x);
					
					int ycnt = 0;
					for (SSVMConstraint constrnt :  instWorkSet.values()) {
						GRBLinExpr exprLeft  = new GRBLinExpr();
						GRBLinExpr exprRight  = new GRBLinExpr();
						
						// right
						exprRight.addConstant(constrnt.loss);
						exprRight.addTerm(-1, xslack);//svars[j]);
						
						// left
						IFeatureVector deltaPhi = constrnt.goldMinusPredFeatures;
						int[] phiIndices = deltaPhi.getIndices();
						float[] phiVals  = deltaPhi.getValues();
						assert(phiIndices.length == phiVals.length);
						for (int k = 0; k < phiIndices.length; k++) {
							int widx   = phiIndices[k];
							float ceff = phiVals[k];
							//System.err.println(widx + " " + wvars.length);
							GRBVar wvar = wvars[widx];
							exprLeft.addTerm(ceff, wvar);
						}
						
					    String ename = "c_" + String.valueOf(j) + "_" + String.valueOf(optCallCnt) + "_" + String.valueOf(ycnt);
					    model.addConstr(exprLeft, GRB.GREATER_EQUAL, exprRight, ename);
					    
					    ycnt++;
					}
					
				}
				

			}
			

			//// 4. Solve
			// Optimize model
			model.optimize();
		      
			//// 5. Output
			modelToWeight(model,w);

		      System.out.println(x.get(GRB.StringAttr.VarName)
		                         + " " +x.get(GRB.DoubleAttr.X));
		      System.out.println(y.get(GRB.StringAttr.VarName)
		                         + " " +y.get(GRB.DoubleAttr.X));
		      System.out.println(z.get(GRB.StringAttr.VarName)
		                         + " " +z.get(GRB.DoubleAttr.X));

		      System.out.println("Obj: " + model.get(GRB.DoubleAttr.ObjVal) + " " +
		                         obj.getValue());
		      System.out.println();


		      //model.dispose();

		} catch (GRBException e) {
			System.out.println("Error code: " + e.getErrorCode() + ". " + e.getMessage());
		}
		

	}
	
	public void modelToWeight(GRBModel mdl, WeightVector w) {
		try {
			for (int i = 0; i < wvars.length; i++) {
				double wi = wvars[i].get(GRB.DoubleAttr.X);
				w.setElement(i, (float)wi);
			}
		} catch (GRBException e) {
			e.printStackTrace();
		}
	}
*/
	public static void main(String[] args) {
		// TODO Auto-generated method stub

	}
	
	public void doSolveQP(SLProblem problem, SSVMWorkSet increamentalWorkset, WeightVector w) {
		
	}
}
