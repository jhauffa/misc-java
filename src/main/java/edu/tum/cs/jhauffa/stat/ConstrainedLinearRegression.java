package edu.tum.cs.jhauffa.stat;

import java.util.Arrays;
import java.util.logging.Logger;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.QRDecomposition;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.SingularValueDecomposition;

public class ConstrainedLinearRegression {

	private static Logger logger = Logger.getLogger(ConstrainedLinearRegression.class.getName());

	/**
	 * Multiple linear regression can be formulated as matrix inversion, as demonstrated by Alpaydın in "Introduction to
	 * Machine Learning", 2nd edition, pp. 103: Given a m x n matrix x of n-dimensional inputs and a m x 1 matrix
	 * (column vector) r of outputs, computes a n x 1 matrix of weights w, so that (x^T * x) * w = x^T * r. x^T is the
	 * transpose of x. Solving this equation for w yields w = (x^T * x)^-1 * x^T * r, where x^-1 is the inverse of x.
	 * To obtain a unique result in case of an under- or overdetermined system of linear equations, we compute the
	 * Moore–Penrose pseudoinverse (see http://en.wikipedia.org/wiki/Moore–Penrose_pseudoinverse) by means of Singular
	 * Value Decomposition.
	 * 
	 * n := dimension of the regression problem; number of variables
	 * m := number of data instances to learn from
	 */
	public static RealMatrix solveMultivariate(RealMatrix x, RealMatrix r) {
		if ((x.getRowDimension() != r.getRowDimension()) || (r.getColumnDimension() != 1))
			throw new IllegalArgumentException("invalid input matrices");

		RealMatrix xt = x.transpose();
		RealMatrix a = xt.multiply(x);
		RealMatrix b = xt.multiply(r);
		SingularValueDecomposition svd = new SingularValueDecomposition(a);
		return svd.getSolver().solve(b);
	}

	private static final double epsilon = 0.000001;	// appears to be sufficient

	/**
	 * Implements the "non-negative least squares" (NNLS) algorithm by Lawson and Hanson, "Solving Least Squares
	 * Problems" (1995). ch. 23. A review of this and related algorithms can be found in Chen and Plemmons,
	 * "Nonnegativity constraints in numerical analysis" (http://www.wfu.edu/~plemmons/papers/nonneg.pdf).
	 */
	public static RealMatrix solveMultivariateNN(RealMatrix x, RealMatrix r) {
		int n = x.getColumnDimension();
		boolean[] isActive = new boolean[n];
		Arrays.fill(isActive, true);
		RealMatrix result = MatrixUtils.createRealMatrix(n, 1);

		/*
		 * According to Lawson and Hanson, the outer loop will typically run 1/2*n times, where n is the number of
		 * coefficients. Choose a larger limit to ensure that at least we get a somewhat reasonable solution within a
		 * somewhat reasonable amount of time. 
		 */
		int numIter = 0;
		while (numIter++ < (2 * x.getColumnDimension())) {
			RealMatrix f = r.subtract(x.multiply(result));
			RealMatrix w = x.transpose().multiply(f);

			int maxWIdx = -1;
			double maxW = 0.0;
			for (int i = 0; i < n; i++) {
				if (isActive[i] && (w.getEntry(i, 0) > maxW)) {
					maxW = w.getEntry(i, 0);
					maxWIdx = i;
				}
			}
			if (maxW <= 0.0)
				break;
			isActive[maxWIdx] = false;

			boolean allPassivePositive;
			do {
				RealMatrix xp = x.copy();
				for (int i = 0; i < n; i++) {
					if (isActive[i])
						for (int j = 0; j < xp.getRowDimension(); j++)
							xp.setEntry(j, i, 0.0);
				}
				RealMatrix z = solveMultivariate(xp, r);
	
				allPassivePositive = true;
				for (int i = 0; i < n; i++) {
					if (isActive[i])
						z.setEntry(i, 0, 0.0);
					else if (z.getEntry(i, 0) <= 0.0)
						allPassivePositive = false;
				}
				if (!allPassivePositive) {
					double minValue = Double.MAX_VALUE;
					for (int i = 0; i < n; i++) {
						if (!isActive[i] && (z.getEntry(i, 0) <= 0.0)) {
							double value = result.getEntry(i, 0) / (result.getEntry(i, 0) - z.getEntry(i, 0));
							if (value < minValue)
								minValue = value;
						}
					}
					z = z.subtract(result);
					z = z.scalarMultiply(minValue);
					result = result.add(z);
		
					for (int i = 0; i < n; i++)
						if (!isActive[i] && (result.getEntry(i, 0) < epsilon)) {
							result.setEntry(i, 0, 0.0);
							isActive[i] = true;
						}
				} else
					result = z;
			} while (!allPassivePositive);
		}
		return result;
	}

	/**
	 * Same as {@link #solveMultivariate(RealMatrix, RealMatrix)}, but with the additional constraint that the
	 * regression coefficients sum to 1.
	 * Implements a special case of the algorithm for solving linear least squares with linear equality constraints
	 * (LSE) as described in Lawson and Hanson, "Solving Least Squares Problems" (1995), ch. 20.
	 */
	public static RealMatrix solveMultivariateS1(RealMatrix x, RealMatrix r) {
		if (x.getRowDimension() < x.getColumnDimension())
			throw new IllegalArgumentException("need more data points than coefficients");

		RealMatrix c = MatrixUtils.createRealMatrix(x.getColumnDimension(), 1);
		for (int i = 0; i < x.getColumnDimension(); i++)
			c.setEntry(i, 0, 1.0);
		QRDecomposition qr = new QRDecomposition(c);
		RealMatrix k = qr.getQ();
		double y1 = 1.0 / qr.getR().getEntry(0, 0);

		RealMatrix e = x.multiply(k);
		RealMatrix e1 = e.getSubMatrix(0, e.getRowDimension() - 1, 0, 0);
		RealMatrix e2 = e.getSubMatrix(0, e.getRowDimension() - 1, 1, e.getColumnDimension() - 1);
		e1 = e1.scalarMultiply(y1);
		RealMatrix f = r.subtract(e1);

		RealMatrix y2 = solveMultivariate(e2, f);
		double[][] yData = new double[y2.getRowDimension() + 1][1];
		yData[0][0] = y1;
		for (int i = 1; i < yData.length; i++)
			yData[i][0] = y2.getEntry(i - 1, 0);
		RealMatrix y = MatrixUtils.createRealMatrix(yData);
		return k.multiply(y);
	}

	/**
	 * Same as {@link #solveMultivariateS1(RealMatrix, RealMatrix)}, but with the additional constraint that the
	 * regression coefficients must not be negative.
	 * 
	 * Outline of the algorithm (referencing chapter 23 of Lawson and Hanson):
	 * - starting with LSI problem of minimizing |Ex - f| subject to Gx >= h, where G = I and h = (0 ... 0)^T, with the
	 *   added equality constraint Cx = d, where C = (1 ... 1) and d = 1; the inequality constraint causes the solution
	 *   vector to be non-negative, and the equality constraint causes its components to sum to 1
	 * - the problem is transformed to an equivalent LSI problem without equality constraints using the first method of
	 *   section 6
	 * - the LSI problem is transformed to an equivalent LDP problem as described in section 5
	 * - the LDP problem is solved using the algorithm of section 4, which invokes the NNLS algorithm given in section 3
	 * - back-transformation to obtain the solution to LSI via equations 23.39 and 23.36
	 * - back-transformation to obtain the solution to LSI with equality constraint via equation 23.42
	 */
	public static RealMatrix solveMultivariateNNS1(RealMatrix x, RealMatrix r) {
		// eliminate equality constraint (sum to 1) by transformation
		RealMatrix c = MatrixUtils.createRealMatrix(x.getColumnDimension(), 1);
		for (int i = 0; i < x.getColumnDimension(); i++)
			c.setEntry(i, 0, 1.0);
		QRDecomposition qr = new QRDecomposition(c);
		RealMatrix k = qr.getQ();
		double y1 = 1.0 / qr.getR().getEntry(0, 0);

		RealMatrix e = x.multiply(k);
		RealMatrix e1 = e.getSubMatrix(0, e.getRowDimension() - 1, 0, 0);
		RealMatrix e2 = e.getSubMatrix(0, e.getRowDimension() - 1, 1, e.getColumnDimension() - 1);
		e1 = e1.scalarMultiply(y1);
		RealMatrix f = r.subtract(e1);

		// transform remaining inequality constraints (coefficients >= 0)
		// G = I => G1 = K1 etc.
		RealMatrix k1 = k.getSubMatrix(0, k.getRowDimension() - 1, 0, 0);
		k1 = k1.scalarMultiply(-1.0 * y1);	// since h = (0 ... 0)^T
		RealMatrix k2 = k.getSubMatrix(0, k.getRowDimension() - 1, 1, k.getColumnDimension() - 1);

		// got LSI: |e2y2 - f|, k2y2 >= k1 -> transform to equivalent LDP problem
		SingularValueDecomposition svd = new SingularValueDecomposition(e2);
		RealMatrix rInv = svd.getS();
		double tol = MathUtil.computeSingularValueThreshold(e2, rInv);
		int numBad = 0;
		for (int i = 0; i < rInv.getColumnDimension(); i++) {
			double v = rInv.getEntry(i, i);
			if (v > tol)
				rInv.setEntry(i, i, 1.0 / v);
			else {
				rInv.setEntry(i, i, 0.0);
				numBad++;
			}
		}
		if (numBad > 0) {
			logger.warning("matrix is ill-conditioned, number of negligible singular values = " + numBad);
		}
		RealMatrix kk = svd.getV();
		RealMatrix ldpG = k2.multiply(kk.multiply(rInv));
		RealMatrix svdU = svd.getU();
		RealMatrix q1 = svdU.getSubMatrix(0, svdU.getRowDimension() - 1, 0, e2.getColumnDimension() - 1);
		RealMatrix f1 = q1.transpose().multiply(f);
		k1 = k1.subtract(ldpG.multiply(f1));

		// actually solve LDP: minimize |z| with constraint ldpG * z >= k1
		int rows = ldpG.getColumnDimension() + 1;
		double[][] ldpEData = new double[rows][ldpG.getRowDimension()];
		for (int i = 0; i < ldpEData.length - 1; i++)
			for (int j = 0; j < ldpEData[i].length; j++)
				ldpEData[i][j] = ldpG.getEntry(j, i);
		for (int i = 0; i < ldpEData[rows - 1].length; i++)
			ldpEData[rows - 1][i] = k1.getEntry(i, 0);
		RealMatrix ldpE = MatrixUtils.createRealMatrix(ldpEData);
		RealMatrix ldpF = MatrixUtils.createRealMatrix(rows, 1);
		ldpF.setEntry(rows - 1, 0, 1.0);
		RealMatrix u = solveMultivariateNN(ldpE, ldpF);
		RealMatrix ldpR = ldpE.multiply(u);
		ldpR = ldpR.subtract(ldpF);
		RealMatrix z = MatrixUtils.createRealMatrix(rows - 1, 1);
		for (int i = 0; i < (rows - 1); i++)
			z.setEntry(i, 0, -ldpR.getEntry(i, 0) / ldpR.getEntry(rows - 1, 0));

		// re-transform to obtain solution of inequality constrained regression
		z = z.add(f1);
		RealMatrix y2 = kk.multiply(rInv.multiply(z));

		// re-transform to obtain solution of equality constrained regression
		double[][] yData = new double[y2.getRowDimension() + 1][1];
		yData[0][0] = y1;
		for (int i = 1; i < yData.length; i++)
			yData[i][0] = y2.getEntry(i - 1, 0);
		RealMatrix y = MatrixUtils.createRealMatrix(yData);
		RealMatrix result = k.multiply(y);

		// Due to limited numerical precision, coefficients may be very small negative values; perform sanity check
		// and round to zero.
		for (int i = 0; i < result.getRowDimension(); i++) {
			double curCoeff = result.getEntry(i, 0);
			if (curCoeff < 0.0) {
				if (-curCoeff > epsilon) {
					throw new RuntimeException("got negative coefficient in NNS1 regression: " + curCoeff);
				}
				result.setEntry(i, 0, 0.0);
			}
		}
		return result;
	}

	/**
	 * Computes the normalized error from the matrix x and vector r used to perform the regression, and the resulting
	 * weight/coefficient vector w. The normalized error is defined as the sum of squared differences between the
	 * actual and the predicted output.
	 */
	public static double computeNormalizedError(RealMatrix x, RealMatrix r, RealMatrix w) {
		RealMatrix err = r.subtract(x.multiply(w));
		double sumSquaredDiff = 0.0;
		int n = err.getRowDimension();
		for (int i = 0; i < n; i++)
			sumSquaredDiff += Math.pow(err.getEntry(i, 0), 2.0);
		n -= x.getColumnDimension();	// subtract number of coefficients to get degrees of freedom for error
		return sumSquaredDiff / n;
	}

	public static double computeR2(RealMatrix x, RealMatrix r, RealMatrix w) {
		RealMatrix err = r.subtract(x.multiply(w));
		double sumSquaredDiff = 0.0;
		int n = err.getRowDimension();
		for (int i = 0; i < n; i++)
			sumSquaredDiff += Math.pow(err.getEntry(i, 0), 2.0);

		double avgR = 0.0;
		for (int i = 0; i < n; i++)
			avgR += r.getEntry(i, 0);
		avgR /= n;
		double sumSquaredVar = 0.0;
		for (int i = 0; i < n; i++)
			sumSquaredVar += Math.pow(r.getEntry(i, 0) - avgR, 2.0);

		return 1.0 - (sumSquaredDiff / sumSquaredVar);
	}

	public static double computeAdjustedR2(RealMatrix x, RealMatrix r, RealMatrix w, boolean hasIntercept) {
		double r2 = computeR2(x, r, w);
		int n = x.getRowDimension();
		int p = x.getColumnDimension();
		double a = (double) (n - (hasIntercept ? 1 : 0)) / (n - p);
		return 1.0 - ((1.0 - r2) * a);
	}

}
