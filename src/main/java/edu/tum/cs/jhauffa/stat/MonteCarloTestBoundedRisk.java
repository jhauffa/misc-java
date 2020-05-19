package edu.tum.cs.jhauffa.stat;

/**
 * Implementation of Gandy's algorithm for computing a sequence of bounds to be used for early exit in Monte Carlo
 * hypothesis tests. See Gandy, 2009, "Sequential Implementation of Monte Carlo Tests with Uniformly Bounded Resampling
 * Risk".
 */
public class MonteCarloTestBoundedRisk {

	private static final double defaultEps = 1E-3;
	private static final int defaultK = 1000;

	private final double eps;
	private final int k;

	public MonteCarloTestBoundedRisk() {
		this(defaultEps, defaultK);
	}

	public MonteCarloTestBoundedRisk(double eps, int k) {
		this.eps = eps;
		this.k = k;
	}

	private double spending(int n) {
		return (eps * n) / (n + k);
	}

	public int[][] computeBounds(int n, double level) {
		double errL = 0.0, errU = 0.0;

		double[] sp = new double[n];
		for (int i = 0; i < n; i++)
			sp[i] = spending(i + 1);

		double[] p0 = new double[n + 1];
		p0[0] = 1.0 - level;
		p0[1] = level;

		int L = -1;
		int U = 2;
		int[][] bounds = new int[2][n];
		bounds[0][0] = L;
		bounds[1][0] = U;

		for (int i = 1; i < n; i++) {
			p0[U] = p0[U - 1] * level;
			for (int j = U - 1; j >= L + 2; j--)
				p0[j] = p0[j] * (1.0 - level) + p0[j - 1] * level;
			p0[L + 1] *= 1.0 - level;

			while (p0[U] + errU <= sp[i]) {
				errU += p0[U];
				U--;
			}
			while (p0[L + 1] + errL <= sp[i]) {
				errL += p0[L + 1];
				L++;
			}
			U++;

			bounds[0][i] = L;
			bounds[1][i] = U;
		}

		return bounds;
	}

}
