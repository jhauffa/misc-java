package edu.tum.cs.jhauffa.stat;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

public class MathUtilTest {

	private static final double tol = 1E-5;

	@Test
	public void testGiniCoefficient() {
		assertTrue(Double.isNaN(MathUtil.giniCoefficient(new double[][] {}, false)));
		assertEquals(0.0, MathUtil.giniCoefficient(new double[][] { { 100.0, 10.0 } }, false), tol);
		assertEquals(0.5, MathUtil.giniCoefficient(new double[][] { { 0.0, 1.0 }, { 100.0, 1.0 } }, false), tol);
		assertEquals(0.8, MathUtil.giniCoefficient(new double[][] { { 0.0, 4.0 }, { 100.0, 1.0 } }, false), tol);
		assertEquals(0.8, MathUtil.giniCoefficient(new double[][] { { 100.0, 100.0 }, { 0.0, 400.0 } }, false), tol);
	}

}
