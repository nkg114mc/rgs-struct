package search;

import java.util.Random;

public class ZobristKeys {
	
	int[][] zobristHashKeys;
	
	public ZobristKeys(int maxNmention, int maxNvalue) {
		initialZobrise(maxNmention, maxNvalue);
	}

	public void initialZobrise(int maxNment, int maxNval) {
		Random rnd = new Random();
		zobristHashKeys = new int[maxNment][maxNval];
		int i, j;
		for (i = 0; i < maxNment; i++) {
			for (j = 0; j < maxNval; j++) {
				zobristHashKeys[i][j] = rnd.nextInt(Integer.MAX_VALUE);
			}
		}
	}
	
	public int[][] getZbKeys() {
		return zobristHashKeys;
	}
}
