package sequence.twitterpos;

import general.AbstractLabelSet;

public class TwitterPosLabelSet extends AbstractLabelSet  {

	private static final String[] TWITTER_POS_TAGS = {
			"N",
			"O",
			"S",
			"^",
			"Z",
			"L",
			"M",
			"V",
			"A",
			"R",
			"!",
			"D",
			"P",
			"&",
			"T",
			"X",
			"Y",
			"#",
			"@",
			"~",
			"U",
			"E",
			"$",
			",",
			"G"
	};

	public String[] getLabels() {
		return TWITTER_POS_TAGS;
	}
}
