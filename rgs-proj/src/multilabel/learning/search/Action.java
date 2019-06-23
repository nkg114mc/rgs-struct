package multilabel.learning.search;

public class Action {
	
	public int position = -1;
	public int newValue = -1;
	public int oldValue = -1;
	
	public Action(int pos, int newv) {
		position = pos;
		newValue = newv;
	}

	public void print() {
		System.out.println("Change at " + position + " to " + newValue);
	}
}
