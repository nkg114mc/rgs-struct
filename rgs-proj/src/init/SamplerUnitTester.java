package init;

import java.util.Arrays;
import java.util.Random;

public class SamplerUnitTester {

	public static void main(String[] args) {
		
		int domainSize = 45;
		
		
		System.out.println("===NonUniform===");
		double[] p = getRandomProbs(domainSize);
		test(p, 10);
		test(p, 100);
		test(p, 1000);
		test(p, 10000);
		test(p, 100000);
		
		System.out.println("//////////////");
		System.out.println("===Uniform===");
		testUniform(domainSize, 10);
		testUniform(domainSize, 100);
		testUniform(domainSize, 1000);
		testUniform(domainSize, 10000);
		testUniform(domainSize, 100000);
		
		System.out.println("Done.");
	}
	
	public static double[] getRandomProbs(int valNum) {
		Random rnd = new Random();
		double[] result = new double[valNum];
		double sum = 0;
		for (int i = 0; i < valNum; i++) {
			result[i] = rnd.nextDouble();
			sum += result[i];
		}
		for (int i = 0; i < valNum; i++) {
			result[i] /= sum;
		}
		return result;
	}
	
	public static void test(double[] probs, int numSamp) {
		
		Random rnd = new Random();
		double total = 0;
		double[] hist = new double[probs.length]; 
		double[] ratio = new double[probs.length]; 
		Arrays.fill(hist, 0);
		
		
		for (int i = 0; i < numSamp; i++ ) {
			total += 1;
			int idx = SeqSamplingRndGenerator.sampleWithProbs(probs, rnd);
			hist[idx] += 1;
		}
		
		Arrays.fill(ratio, 0);
		for (int j = 0; j < probs.length; j++) {
			ratio[j] = hist[j] / total;
		}
		
		/////////////////
		// print histogram
		
		System.out.println("==== SampleCnt " + numSamp + "====");
		for (int j = 0; j < probs.length; j++) {
			System.out.println("Index " + j + ": " +  ratio[j] + " | " + probs[j]);
		}
		System.out.println("========End========");
	}

	public static void testUniform(int domainSz, int numSamp) {
		
		Random rnd = new Random();
		double total = 0;
		double[] hist = new double[domainSz]; 
		double[] ratio = new double[domainSz]; 
		Arrays.fill(hist, 0);
		
		
		for (int i = 0; i < numSamp; i++ ) {
			total += 1;
			int idx = UniformRndGenerator.getValueIndexUniformly(domainSz, rnd);//SeqSamplingRndGenerator.sampleWithProbs(probs, rnd);
			hist[idx] += 1;
		}
		
		Arrays.fill(ratio, 0);
		for (int j = 0; j < domainSz; j++) {
			ratio[j] = hist[j] / total;
		}
		
		/////////////////
		// print histogram
		
		System.out.println("==== SampleCnt " + numSamp + "====");
		for (int j = 0; j < domainSz; j++) {
			System.out.println("Index " + j + ": " +  ratio[j]);
		}
		System.out.println("========End========");
	}
	
}
