package edu.tum.cs.jhauffa.stat;

import java.util.Arrays;
import java.util.Comparator;

import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.util.Precision;

public class MathUtil {

	private static final double MIN_DOUBLE_EPSILON = 0x1.0p-52;	/** machine epsilon */

	/**
	 * This code has been lifted from class SingularValueDecomposition (Apache Math), since its member variable "tol" is
	 * private. It computes a threshold for identifying non-negligible singular values. See the documentation of
	 * SingularValueDecomposition.getRank for details.
	 */
	public static double computeSingularValueThreshold(RealMatrix m, RealMatrix s) {
		int maxDim = Math.max(m.getRowDimension(), m.getColumnDimension());
		return Math.max(maxDim * s.getEntry(0, 0) * MIN_DOUBLE_EPSILON, Math.sqrt(Precision.SAFE_MIN));
	}

	/**
	 * Computes the Gini coefficient of an array of observation data.
	 * @param rawCounts An n x 2 matrix. The first column contains the observed values, the second column contains the
	 * 	frequency of observation. The frequency can be absolute or relative.
	 * @param isSorted If true, the observed values are assumed to be in ascending order.
	 */
	public static double giniCoefficient(double[][] rawCounts, boolean isSorted) {
		if (!isSorted) {
			Arrays.sort(rawCounts, new Comparator<double[]>() {
				@Override
				public int compare(double[] o1, double[] o2) {
					return Double.compare(o1[0], o2[0]);
				}
			});
		}

		double sum = 0.0;
		for (double[] v : rawCounts)
			sum += v[1];
		for (double[] v : rawCounts)
			v[1] /= sum;
		return giniCoefficient(rawCounts);
	}

	/**
	 * Computes the Gini coefficient of an array of observation data.
	 * @param normCounts An n x 2 matrix. The first column contains the observed values, the second column contains the
	 * 	relative frequency of observation. Values of the first column must be in ascending order and the values of the
	 * 	second column must sum to one.
	 */
	public static double giniCoefficient(double[][] normCounts) {
		double sPrev = 0.0, sCur = 0.0, sum = 0.0;
		for (double[] v : normCounts) {
			sCur += v[0] * v[1];
			sum += v[1] * (sPrev + sCur);
			sPrev = sCur;
		}
		return 1.0 - (sum / sCur);
	}

}
