package imgseg;

import java.util.ArrayList;
import java.util.List;

import imgcnn.ActualPixel;
import imgcnn.SuperPixelEdge;

public class ImageSuperPixel {
	
	private int id;
	
	private int goldLabel;
	private int pixelCnt;
	
	public double features[][]; //[class][ith-classifier]
	public int[] neighours;
	
	public int hwsegIndex; // for featurizing only
	
	
	
	private List<ActualPixel> pixels;
	private List<SuperPixelEdge> edges;
	
	public double[] centroid;
	
	
	public ImageSuperPixel(int index) {
		id = index;
		pixels = null;
	}
	
	public int getId() {
		return id;
	}
	
	public void setLabel(int lb) {
		goldLabel = lb;
	}
	
	public int getLabel() {
		return goldLabel;
	}
	
	
	public void setPixCnt(int cnt) {
		pixelCnt = cnt;
	}
	
	public int getPixCnt() {
		return pixelCnt;
	}

	public void setZMap(List<ActualPixel> pixs) {
		pixels = pixs;
		pixelCnt = pixs.size();
	}
	
	public List<ActualPixel> getZMap() {
		return pixels;
	}
	
	public List<SuperPixelEdge> getAdjEdges() {
		return edges;
	}
	
	public void setAdjEdges(List<SuperPixelEdge> edgs) {
		edges = edgs;
	}
	
	public int getAdjSupixCnt() {
		if (edges == null) {
			return 0;
		}
		return edges.size();
	}
}
