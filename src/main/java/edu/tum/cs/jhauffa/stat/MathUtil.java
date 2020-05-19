package edu.tum.cs.jhauffa.stat;

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

}
