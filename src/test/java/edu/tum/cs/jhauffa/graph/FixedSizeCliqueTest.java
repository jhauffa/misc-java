package edu.tum.cs.jhauffa.graph;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashSet;
import java.util.Random;
import java.util.Set;

import org.junit.Test;

import edu.uci.ics.jung.graph.UndirectedGraph;
import edu.uci.ics.jung.graph.UndirectedSparseGraph;

public class FixedSizeCliqueTest {

	private static final long seed = 123787L;
	private static final int minCliqueSize = 4;
	private static final int maxCliqueSize = 10;
	private static final int numCliques = 3;

	private static UndirectedGraph<Integer, Integer> generateRandomGraph(Random rng, int numCliques, int cliqueSize,
			Collection<Set<Integer>> cliques) {
		UndirectedGraph<Integer, Integer> g = new UndirectedSparseGraph<Integer, Integer>();
		int numNodes = 2 * numCliques * cliqueSize;
		ArrayList<Integer> p = new ArrayList<Integer>(numNodes);
		for (int i = 0; i < numNodes; i++) {
			p.add(i);
			g.addVertex(i);
		}
		Collections.shuffle(p, rng);
		int edgeIdx = 0;

		// in each clique, connect each node with every other node
		int baseIdx = 0;
		for (int i = 0; i < numCliques; i++) {
			Set<Integer> cliqueNodes = new HashSet<Integer>();
			for (int j = 0; j < cliqueSize; j++)
				cliqueNodes.add(p.get(baseIdx + j));
			cliques.add(cliqueNodes);

			for (int j = 1; j < cliqueSize; j++) {
				for (int k = 0; k < j; k++)
					g.addEdge(edgeIdx++, p.get(baseIdx + j), p.get(baseIdx + k));
			}
			baseIdx += cliqueSize;
		}

		// randomly connect the remaining nodes
		for (int i = baseIdx; i < numNodes; i++) {
			for (int j = 0; j < i; j++) {
				if (rng.nextDouble() < 0.2)
					g.addEdge(edgeIdx++, p.get(i), p.get(j));
			}
		}
		return g;
	}

	private static void checkCliques(Collection<Set<Integer>> refCliques, Collection<Collection<Integer>> cliques) {
		assertTrue(refCliques.size() <= cliques.size());
		int numFound = 0;
		for (Collection<Integer> clique : cliques) {
nextRefClique:
			for (Set<Integer> refClique : refCliques) {
				if (refClique.size() != clique.size())
					continue nextRefClique;
				for (Integer node : clique)
					if (!refClique.contains(node))
						continue nextRefClique;
				numFound++;
				break;
			}
		}
		assertEquals(refCliques.size(), numFound);
	}

	@Test
	public void testCliqueFinding() {
		Random rng = new Random(seed);
		for (int i = minCliqueSize; i <= maxCliqueSize; i++) {
			Collection<Set<Integer>> cliques = new ArrayList<Set<Integer>>(numCliques);
			UndirectedGraph<Integer, Integer> g = generateRandomGraph(rng, numCliques, i, cliques);

			FixedSizeClique<Integer, Integer> cliqueFinder = new FixedSizeClique<Integer, Integer>(g, false);
			checkCliques(cliques, cliqueFinder.getAllFixedSizeCliques(i));
			cliqueFinder = new FixedSizeClique<Integer, Integer>(g, true);
			checkCliques(cliques, cliqueFinder.getAllFixedSizeCliques(i));
		}
	}

}
