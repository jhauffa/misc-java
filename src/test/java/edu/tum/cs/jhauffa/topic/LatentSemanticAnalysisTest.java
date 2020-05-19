package edu.tum.cs.jhauffa.topic;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;

import org.junit.Test;

public class LatentSemanticAnalysisTest {

	private static String[][] trainData = {
		{ "dog", "cat", "mouse" },
		{ "hamster", "mouse", "sushi" },
		{ "dog", "monster", "monster" }
	};
	// reference values have been verified against the R package "lsa"
	private static double[][] refConceptDocTrain = {
		{ -0.100, -0.146,  0.984 },
		{ -0.026, -0.988, -0.149 },
		{ -0.995,  0.040, -0.095 }
	};
	private static double[][] refTermConceptTrain = {
		{ -0.248, -0.027,  0.295 },
		{ -0.061, -0.100,  0.884 },
		{ -0.028, -0.285,  0.277 },
		{ -0.016, -0.673, -0.134 },
		{ -0.016, -0.673, -0.134 },
		{ -0.966,  0.044, -0.135 },
		{  0.000,  0.000,  0.000 }
	};
	private static String[][] queryData = {
		{ "cat", "mouse", "mouse" },
		{ "nothing", "mouse", "monster" },
		{ "cat", "monster", "monster" }
	};
	private static double[][] refConceptDocQuery = {
		{ -0.048, -0.181,  0.940 },
		{ -0.599, -0.042, -0.030 },
		{ -0.976, -0.021,  0.602 }
	};

	private static class Index {
		private final Map<String, Integer> wordIndex = new HashMap<String, Integer>();

		public int addWord(String s) {
			Integer word = wordIndex.get(s);
			if (word == null) {
				word = wordIndex.size();
				wordIndex.put(s, word);
			}
			return word;
		}

		public int getNumUniqueWords() {
			return wordIndex.size();
		}
	}

	private static class BasicCorpus implements Corpus {
		private final Index index;
		private final Collection<Document> docs = new ArrayList<Document>();

		public BasicCorpus(Index index) {
			this.index = index;
		}

		public void addDocument(String[] doc) {
			Collection<Integer> words = new ArrayList<Integer>(doc.length);
			for (String s : doc)
				words.add(index.addWord(s));
			docs.add(new BasicDocument(words));
		}

		@Override
		public Iterator<Document> iterator() {
			return docs.iterator();
		}

		@Override
		public int size() {
			return docs.size();
		}

		@Override
		public int getNumUniqueWords() {
			return index.getNumUniqueWords();
		}
	}

	private static class BasicDocument implements Document {
		private final Collection<Integer> words;

		public BasicDocument(Collection<Integer> words) {
			this.words = words;
		}

		@Override
		public Iterator<Integer> iterator() {
			return words.iterator();
		}

		@Override
		public int size() {
			return words.size();
		}
	}

	private static void compareMatrix(double[][] expected, double[][] actual, double epsilon) {
		assertEquals(expected.length, actual.length);
		for (int i = 0; i < expected.length; i++) {
			assertEquals(expected[i].length, actual[i].length);
			for (int j = 0; j < expected[i].length; j++)
				assertTrue(Math.abs(expected[i][j] - actual[i][j]) <= epsilon);
		}
	}

	@Test
	public void testLSA() throws Exception {
		// create both corpora ahead of time so that the bag of words contains all words
		Index bagOfWords = new Index();
		BasicCorpus trainCorpus = new BasicCorpus(bagOfWords);
		for (String[] doc : trainData)
			trainCorpus.addDocument(doc);
		BasicCorpus queryCorpus = new BasicCorpus(bagOfWords);
		for (String[] doc : queryData)
			queryCorpus.addDocument(doc);

		LatentSemanticAnalysis lsa = new LatentSemanticAnalysis(3);
		double[][] conceptDocTrain = lsa.train(trainCorpus);
		compareMatrix(refConceptDocTrain, conceptDocTrain, 0.001);
		compareMatrix(refTermConceptTrain, lsa.getTermConceptMatrix(), 0.001);
		double[][] conceptDocQuery = lsa.query(queryCorpus);
		compareMatrix(refConceptDocQuery, conceptDocQuery, 0.001);

		File tempFile = File.createTempFile("lsa-test-", ".ser");
		tempFile.deleteOnExit();
		ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(tempFile));
		try {
			out.writeObject(lsa);
		} finally {
			out.close();
		}
		ObjectInputStream in = new ObjectInputStream(new FileInputStream(tempFile));
		try {
			lsa = (LatentSemanticAnalysis) in.readObject();
		} finally {
			in.close();
		}

		double[][] conceptDocQuery2 = lsa.query(queryCorpus);
		compareMatrix(conceptDocQuery, conceptDocQuery2, 0.0);
	}

}
