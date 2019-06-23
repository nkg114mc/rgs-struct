package horse256;

public class HorseLabel {
	
	public static final String[] FOREBACK_GROUND_DOMAIN = { "0", "1" };
	
	public static int GtColorToLabelIndex(int rgb) {
		if (rgb == 0) {
			return 0;
		} else {
			return 1;
		}
	}

}
