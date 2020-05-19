package edu.tum.cs.jhauffa.stat;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.util.Random;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.junit.Test;

public class ConstrainedLinearRegressionTest {

	private static final double[][] dataInput = {
		{ 87, 42 },
		{ 73, 43 },
		{ 66, 44 },
		{ 62, 54 },
		{ 68, 45 },
		{ 92, 46 },
		{ 60, 50 },
		{ 70, 46 },
		{ 71, 54 },
		{ 64, 47 }
	};

	private static final double[][] dataOutput = {
		{ 1 }, { 6 }, { 7 }, { 15 }, { 12 }, { 4 }, { 12 }, { 13 }, { 14 }, { 10 }
	};

	private static final double[] expectedCoefficients = { -0.2409, 0.5644 };
	private static final double[] expectedCoefficientsWithIntercept = { -1.7860, -0.2324, 0.5893 };

	private static final double[][] dataInputConstrained = {
		{ 0.0005, 0.0033, 0.2798, 0.6072, 0.6149 },
		{ 0.0000, 0.1475, 0.0000, 0.0001, 0.0000 },
		{ 0.0000, 0.0000, 0.0000, 0.0000, 0.0000 },
		{ 0.0044, 0.0015, 0.0239, 0.0010, 0.3515 },
		{ 0.0026, 0.0000, 0.1674, 0.0001, 0.0000 },
		{ 0.0661, 0.0044, 0.0294, 0.0000, 0.0008 },
		{ 0.0010, 0.8423, 0.0297, 0.1281, 0.0005 },
		{ 0.9113, 0.0003, 0.0389, 0.0000, 0.0000 },
		{ 0.0000, 0.0006, 0.0011, 0.2259, 0.0322 },
		{ 0.0141, 0.0000, 0.4296, 0.0376, 0.0002 }
	};

	private static final double[][] dataOutputConstrained = {
		{ 0.7154 }, { 0.0136 }, { 0.0585 }, { 0 }, { 0.0976 }, { 0.0061 }, { 0 }, { 0.0077 }, { 0 }, { 0.1013 }
	};

	private static final double[] expectedCoefficientsS1 = { -0.0274, -0.1337, 0.2518, 0.6780, 0.2312 };
	private static final double[] expectedCoefficientsNN = { 0.0000, 0.0000, 0.3081, 0.5940, 0.3141 };
	private static final double[] expectedCoefficientsNNS1 = { 0.0000, 0.0000, 0.1600, 0.5554, 0.2846 };

	private static void evaluateLinearRegression(RealMatrix x, RealMatrix r, RealMatrix w, boolean hasIntercept,
			double[] expCoeff, double expMse, double expR2, double expAdjR2) {
		assertEquals(1, w.getColumnDimension());
		assertEquals(expCoeff.length, w.getRowDimension());
		for (int i = 0; i < expCoeff.length; i++)
			assertEquals(expCoeff[i], w.getEntry(i, 0), 0.0001);
		assertEquals(expMse, Math.sqrt(ConstrainedLinearRegression.computeNormalizedError(x, r, w)), 0.01);
		assertEquals(expR2, ConstrainedLinearRegression.computeR2(x, r, w), 0.0001);
		assertEquals(expAdjR2, ConstrainedLinearRegression.computeAdjustedR2(x, r, w, hasIntercept), 0.0001);
	}

	@Test
	public void testLinearRegression() {
		// unconstrained, without intercept
		RealMatrix x = MatrixUtils.createRealMatrix(dataInput);
		RealMatrix r = MatrixUtils.createRealMatrix(dataOutput);
		RealMatrix w = ConstrainedLinearRegression.solveMultivariate(x, r);
		evaluateLinearRegression(x, r, w, false, expectedCoefficients, 2.13, 0.8144, 0.7680);

		// unconstrained, with intercept
		x = MatrixUtils.createRealMatrix(x.getRowDimension(), x.getColumnDimension() + 1);
		x.setSubMatrix(dataInput, 0, 1);
		for (int i = 0; i < x.getRowDimension(); i++)
			x.setEntry(i, 0, 1.0);
		w = ConstrainedLinearRegression.solveMultivariate(x, r);
		evaluateLinearRegression(x, r, w, true, expectedCoefficientsWithIntercept, 2.28, 0.8149, 0.7620);

		// S1, no intercept
		x = MatrixUtils.createRealMatrix(dataInputConstrained);
		r = MatrixUtils.createRealMatrix(dataOutputConstrained);
		w = ConstrainedLinearRegression.solveMultivariateS1(x, r);
		evaluateLinearRegression(x, r, w, false, expectedCoefficientsS1, 0.1014, 0.8819, 0.7638);

		// NN, no intercept
		w = ConstrainedLinearRegression.solveMultivariateNN(x, r);
		evaluateLinearRegression(x, r, w, false, expectedCoefficientsNN, 0.1063, 0.8702, 0.7403);

		// NNS1, no intercept
		w = ConstrainedLinearRegression.solveMultivariateNNS1(x, r);
		evaluateLinearRegression(x, r, w, false, expectedCoefficientsNNS1, 0.1172, 0.8421, 0.6843);
	}

	private static final long seed = 1337L;
	private static final double errorThreshold = 0.3;
	private static final double maxSumDelta = 1E-14;

	@Test
	public void testConstrainedLinearRegressionProperties() {
		Random rand = new Random(seed);
		for (int i = 0; i < 100; i++) {
			double[][] xData = new double[10][5];
			for (int j = 0; j < 10; j++)
				for (int k = 0; k < 5; k++)
					xData[j][k] = rand.nextDouble();
			RealMatrix x = MatrixUtils.createRealMatrix(xData);
			double[][] rData = new double[10][1];
			for (int j = 0; j < 10; j++)
				rData[j][0] = rand.nextDouble();
			RealMatrix r = MatrixUtils.createRealMatrix(rData);

			// S1
			RealMatrix w = ConstrainedLinearRegression.solveMultivariateS1(x, r);
			double sum = 0.0;
			for (int j = 0; j < 5; j++)
				sum += w.getEntry(j, 0);
			assertEquals(1.0, sum, maxSumDelta);
			double err = ConstrainedLinearRegression.computeNormalizedError(x, r, w);
			assertTrue(err + " >= " + errorThreshold, err < errorThreshold);

			// NN
			w = ConstrainedLinearRegression.solveMultivariateNN(x, r);
			for (int j = 0; j < 5; j++) {
				double c = w.getEntry(j, 0);
				assertTrue(c >= 0.0);
			}
			err = ConstrainedLinearRegression.computeNormalizedError(x, r, w);
			assertTrue(err + " >= " + errorThreshold, err < errorThreshold);

			// NNS1
			w = ConstrainedLinearRegression.solveMultivariateNNS1(x, r);
			sum = 0.0;
			for (int j = 0; j < 5; j++) {
				double c = w.getEntry(j, 0);
				assertTrue(c >= 0.0);
				sum += c;
			}
			assertEquals(1.0, sum, maxSumDelta);
			err = ConstrainedLinearRegression.computeNormalizedError(x, r, w);
			assertTrue(err + " >= " + errorThreshold, err < errorThreshold);
		}
	}

}
