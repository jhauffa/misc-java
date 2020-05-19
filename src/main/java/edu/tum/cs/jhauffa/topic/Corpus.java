package edu.tum.cs.jhauffa.topic;

public interface Corpus extends Iterable<Document> {

	/** @return the number of documents in the corpus. */
	public int size();

	/** @return the number of unique words across all documents. */
	public int getNumUniqueWords();

}
