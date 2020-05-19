package edu.tum.cs.jhauffa.stat;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

import org.junit.Test;

public class MonteCarloTestBoundedRiskTest {

	private static final int[][] refBounds = {
		{ -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
		  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
		  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
		  -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
		{ 2, 3, 4, 5, 6, 6, 6, 7, 7, 8, 8, 8, 9, 9, 9, 9, 10, 10, 10, 10, 11, 11, 11, 12, 12, 12, 12, 12, 13, 13, 13,
		  13, 13, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 16, 16, 16, 16, 17, 17, 17, 17, 17, 17, 18, 18, 18, 18,
		  18, 19, 19, 19, 19, 19, 19, 20, 20, 20, 20, 20, 20, 21, 21, 21, 21, 21, 21, 22, 22, 22, 22, 22, 22, 23, 23,
		  23, 23, 23, 23, 23, 24, 24, 24, 24, 24, 25, 25, 25, 25, 25 }
	};

	@Test
	public void testBounds() {
		MonteCarloTestBoundedRisk mc = new MonteCarloTestBoundedRisk(1E-3, 1000);
		int[][] bounds = mc.computeBounds(100, 0.1);
		assertEquals(2, bounds.length);
		assertArrayEquals(refBounds[0], bounds[0]);
		assertArrayEquals(refBounds[1], bounds[1]);
	}

}
