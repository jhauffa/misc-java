package edu.tum.cs.jhauffa.topic;

public interface Document extends Iterable<Integer> {

	/** @return the number of words in the document. */
	public int size();

}
