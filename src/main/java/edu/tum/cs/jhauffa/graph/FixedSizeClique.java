package edu.tum.cs.jhauffa.graph;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Stack;

import edu.uci.ics.jung.graph.UndirectedGraph;

/**
 * Implements the algorithm "COMPLETE" (second, refined version) from the paper "Arboricity and Subgraph Listing
 * Algorithms" by Chiba and Nishizeki.
 * @param <V> vertex class of graph
 * @param <E> edge class of graph
 */
public class FixedSizeClique<V, E> {

	private final List<V> vertices;
	private final int[][] adjacency;
	private final int[] label;
	private final boolean sortVertices;

	public FixedSizeClique(UndirectedGraph<V, E> graph) {
		this(graph, false);
	}

	public FixedSizeClique(UndirectedGraph<V, E> graph, boolean sortVertices) {
		vertices = new ArrayList<V>(graph.getVertices());
		Map<V, Integer> vertexIndices = new HashMap<V, Integer>();
		for (int i = 0; i < vertices.size(); i++)
			vertexIndices.put(vertices.get(i), i);
		adjacency = new int[vertices.size()][];
		for (int i = 0; i < vertices.size(); i++) {
			Collection<V> neighbors = graph.getNeighbors(vertices.get(i));
			adjacency[i] = new int[neighbors.size()];
			int j = 0;
			for (V neighbor : neighbors)
				adjacency[i][j++] = vertexIndices.get(neighbor);
		}
		label = new int[vertices.size()];
		this.sortVertices = sortVertices;
	}

	private void sortByDegree(int[] index, int[] degree) {
		int minDegree = Integer.MAX_VALUE;
		int maxDegree = 0;
		for (int d : degree) {
			if (d > maxDegree)
				maxDegree = d;
			if (d < minDegree)
				minDegree = d;
		}
		int numBuckets = (maxDegree - minDegree) + 1;
		@SuppressWarnings("unchecked")
		List<Integer>[] buckets = (List<Integer>[]) new List[numBuckets];

		for (int i = 0; i < index.length; i++) {
			int bucketIdx = degree[i] - minDegree;
			if (buckets[bucketIdx] == null)
				buckets[bucketIdx] = new LinkedList<Integer>();
			buckets[bucketIdx].add(index[i]);
		}

		int outIdx = 0;
		for (int i = (numBuckets - 1); i >= 0; i--) {	// insert in order of descending degree
			if (buckets[i] != null) {
				for (Integer idx : buckets[i]) {
					index[outIdx] = idx;
					degree[outIdx] = minDegree + i;
					outIdx++;
				}
			}
		}
	}

	private void findCliques(Collection<Collection<V>> cliques, int size, Stack<Integer> candidates, int[] vertexSubset,
			int[] vertexDegree) {
		if (size == 2) {
			for (int i = 0; i < vertexSubset.length; i++) {
				int v = vertexSubset[i];
				for (int j = 0; j < vertexDegree[i]; j++) {
					int w = adjacency[v][j];
					if (v > w)
						continue;

					Collection<V> clique = new ArrayList<V>(candidates.size() + 2);
					clique.add(vertices.get(v));
					clique.add(vertices.get(w));
					for (Integer c : candidates)
						clique.add(vertices.get(c));
					cliques.add(clique);
				}
			}
		} else {
			if (sortVertices && (vertexSubset.length > 1))
				sortByDegree(vertexSubset, vertexDegree);

			for (int i = 0; i < vertexSubset.length; i++) {
				int v = vertexSubset[i];
				int d = 0;	// cannot use vertexDegree[i] as deletion of v at end of loop may affect degree
				for (int j = 0; j < adjacency[v].length; j++) {
					if (label[adjacency[v][j]] > size)
						break;
					d++;
				}

				int[] vertexSubsetNew = new int[d];
				int[] vertexDegreeNew = new int[d];
				System.arraycopy(adjacency[v], 0, vertexSubsetNew, 0, d);
				for (int w : vertexSubsetNew)
					label[w] = size - 1;
				for (int j = 0; j < vertexSubsetNew.length; j++) {
					int w = vertexSubsetNew[j];
					int swapIdx = 0;
					for (int k = 0; k < adjacency[w].length; k++) {
						int x = adjacency[w][k];
						if (label[x] > size)
							break;
						if (label[x] == (size - 1)) {
							adjacency[w][k] = adjacency[w][swapIdx];
							adjacency[w][swapIdx] = x;
							swapIdx++;
						}
					}
					vertexDegreeNew[j] = swapIdx;
				}
				
				candidates.push(v);
				findCliques(cliques, size - 1, candidates, vertexSubsetNew, vertexDegreeNew);
				candidates.pop();

				for (int w : vertexSubsetNew)
					label[w] = size;
				label[v] = size + 1;
				for (int w : vertexSubsetNew) {
					int srcIdx = -1;
					int dstIdx = adjacency[w].length - 1;
					for (int j = 0; j < adjacency[w].length; j++) {
						int x = adjacency[w][j];
						if (x == v) {
							srcIdx = j;
						} else if (label[x] == (size + 1)) {
							dstIdx = j - 1;
							break;
						}
					}
					adjacency[w][srcIdx] = adjacency[w][dstIdx];
					adjacency[w][dstIdx] = v;
				}
			}
		}
	}

	public Collection<Collection<V>> getAllFixedSizeCliques(int size) {
		if (size < 2)
			throw new IllegalArgumentException(size + " < minimum clique size 2");
		Arrays.fill(label, size);

		Collection<Collection<V>> cliques = new ArrayList<Collection<V>>();
		Stack<Integer> candidates = new Stack<Integer>();
		int[] vertexSubset = new int[vertices.size()];
		int[] vertexDegree = new int[vertices.size()];
		for (int i = 0; i < vertices.size(); i++) {
			vertexSubset[i] = i;
			vertexDegree[i] = adjacency[i].length;
		}
		findCliques(cliques, size, candidates, vertexSubset, vertexDegree);
		return cliques;
	}

}
