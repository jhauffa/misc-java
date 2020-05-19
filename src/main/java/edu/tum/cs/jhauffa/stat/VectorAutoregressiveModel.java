package edu.tum.cs.jhauffa.stat;

import java.util.Arrays;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.SingularValueDecomposition;

public class VectorAutoregressiveModel {

	private final RealMatrix coeff;
	private final RealMatrix errorCovariance;

	public VectorAutoregressiveModel(RealMatrix coeff, RealMatrix errorCovariance) {
		this.coeff = coeff;
		this.errorCovariance = errorCovariance;
	}

	/** fit a VAR(1) model to the data */
	public static VectorAutoregressiveModel fit(double[][] obs) {
		// given observation vectors o0 .. oN, build matrices Y = ( o1 .. oN ) and Z = ( 1  .. 1    )
		//                                                                             ( o0 .. oN-1 )
		double[][] yData = new double[obs.length][obs[0].length - 1];
		for (int i = 0; i < obs.length; i++)
			System.arraycopy(obs[i], 1, yData[i], 0, obs[i].length - 1);
		RealMatrix y = new Array2DRowRealMatrix(yData, false);
		double[][] zData = new double[obs.length + 1][obs[0].length - 1];
		Arrays.fill(zData[0], 1.0);
		for (int i = 0; i < obs.length; i++)
			System.arraycopy(obs[i], 0, zData[i + 1], 0, obs[i].length - 1);
		RealMatrix z = new Array2DRowRealMatrix(zData, false);

		// estimate coefficient matrix B = Y * Z^T * (Z*Z^T)^-1
		RealMatrix zt = z.transpose();
		RealMatrix zzt = z.multiply(zt);
		SingularValueDecomposition svd = new SingularValueDecomposition(zzt);
		RealMatrix zztInv = svd.getSolver().getInverse();
		RealMatrix b = y.multiply(zt.multiply(zztInv));

		// compute error covariances
		RealMatrix e = y.subtract(b.multiply(z));
		int df = obs[0].length - obs.length - 1;	// #obs - #params - 1 = #obs - #dim - 1
		RealMatrix cov = e.multiply(e.transpose()).scalarMultiply(1.0 / df);

		return new VectorAutoregressiveModel(b, cov);
	}

	/** @return coefficient matrix B in concise notation */
	public RealMatrix getCoefficients() {
		return coeff;
	}

	/** @return error covariance matrix of training data */
	public RealMatrix getErrorCovariance() {
		return errorCovariance;
	}

}
