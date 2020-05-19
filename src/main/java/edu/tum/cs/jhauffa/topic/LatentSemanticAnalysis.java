package edu.tum.cs.jhauffa.topic;

import java.io.Serializable;
import java.util.HashMap;
import java.util.Map;
import java.util.logging.Logger;

import org.apache.commons.math3.linear.OpenMapRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.SingularValueDecomposition;

import edu.tum.cs.jhauffa.stat.MathUtil;

/**
 * Performs Latent Semantic Analysis with log entropy weighting.
 */
public class LatentSemanticAnalysis implements Serializable {

	private static Logger logger = Logger.getLogger(LatentSemanticAnalysis.class.getName());
	private static final long serialVersionUID = -9121027710326643366L;
	private static final double LOGBASETWO = Math.log(2.0);

	private final int numDimensions;
	private double[] globalWeight;
	private RealMatrix t, sInv;

	public LatentSemanticAnalysis(int numDimensions) {
		this.numDimensions = numDimensions;
	}

	private Map<Integer, Integer> countWords(Document doc) {
		Map<Integer, Integer> docFreq = new HashMap<Integer, Integer>();
		for (int id : doc) {
			Integer count = docFreq.get(id);
			if (count == null)
				count = 0;
			docFreq.put(id, count + 1);
		}
		return docFreq;
	}

	private double[] computeGlobalWeights(Corpus corpus) {
		int numTerms = corpus.getNumUniqueWords();

		// compute global term frequency for global entropy weighting
		int[] globalFreq = new int[numTerms];
		for (Document doc : corpus)
			for (int id : doc)
				globalFreq[id]++;

		// compute global weights
		double[] globalWeight = new double[numTerms];
		for (Document doc : corpus) {
			Map<Integer, Integer> docFreq = countWords(doc);
			for (Map.Entry<Integer, Integer> e : docFreq.entrySet()) {
				int id = e.getKey();
				double p = (double) e.getValue() / globalFreq[id];
				globalWeight[id] += p * (Math.log(p) / LOGBASETWO);
			}
		}
		for (int i = 0; i < numTerms; i++)
			globalWeight[i] = 1.0 + (globalWeight[i] / (Math.log(corpus.size()) / LOGBASETWO));
		return globalWeight;
	}

	private RealMatrix buildTermDocumentMatrix(Corpus corpus) {
		int numTerms = corpus.getNumUniqueWords();

		// build term-document matrix
		RealMatrix termDocumentMatrix = new OpenMapRealMatrix(numTerms, corpus.size());
		int docIdx = 0;
		for (Document doc : corpus) {
			Map<Integer, Integer> docFreq = countWords(doc);
			for (Map.Entry<Integer, Integer> e : docFreq.entrySet()) {
				int id = e.getKey();
				double termDocumentWeight = globalWeight[id] * (Math.log(e.getValue() + 1) / LOGBASETWO);
				termDocumentMatrix.setEntry(id, docIdx, termDocumentWeight);
			}
			docIdx++;
		}
		return termDocumentMatrix;
	}

	/**
	 * @return the concept-document matrix for the specified corpus
	 */
	public double[][] train(Corpus corpus) {
		// perform truncated SVD on term-document matrix
		globalWeight = computeGlobalWeights(corpus);
		RealMatrix termDocumentMatrix = buildTermDocumentMatrix(corpus);
		SingularValueDecomposition svd = new SingularValueDecomposition(termDocumentMatrix);
		int k = Math.min(Math.min(termDocumentMatrix.getRowDimension(), termDocumentMatrix.getColumnDimension()),
				numDimensions);

		// compute inverse singular values, which will be used for querying; perform basic sanity checking
		sInv = svd.getS().getSubMatrix(0, k - 1, 0, k - 1);
		double tol = MathUtil.computeSingularValueThreshold(termDocumentMatrix, sInv);
		for (int i = 0; i < k; i++) {
			double v = sInv.getEntry(i, i);
			if (v <= tol)
				logger.warning("singular value " + (i + 1) + " below threshold: " + v);
			sInv.setEntry(i, i, 1.0 / v);
		}

		t = svd.getU().getSubMatrix(0, termDocumentMatrix.getRowDimension() - 1, 0, k - 1);
		return svd.getV().getSubMatrix(0, termDocumentMatrix.getColumnDimension() - 1, 0, k - 1).getData();
	}

	public double[][] getTermConceptMatrix() {
		return t.getData();
	}

	/**
	 * @param corpus The specified corpus has to use the same bag of words as the corpus used for training!
	 * @return the concept-document matrix for the specified corpus, given the global term weights, term-concept matrix,
	 * 	and singular values computed by {@link #train(Corpus)}
	 */
	public double[][] query(Corpus corpus) {
		RealMatrix termDocumentMatrix = buildTermDocumentMatrix(corpus);
		return termDocumentMatrix.transpose().multiply(t).multiply(sInv).getData();
	}

}
