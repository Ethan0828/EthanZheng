# Introduction to Spectral Graph Neural Networks

Graph Neural Networks (GNNs) have emerged as the dominant paradigm for learning on graph-structured data. Among the various GNN architectures, **spectral methods** ground the computation in the rigorous theory of graph signal processing.

## The Graph Laplacian

Given a graph $G = (V, E)$ with $n$ nodes, the normalized graph Laplacian is:

$$\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1/2} \mathbf{A} \mathbf{D}^{-1/2}$$

where $\mathbf{A}$ is the adjacency matrix and $\mathbf{D}$ is the degree matrix. The Laplacian admits an eigendecomposition:

$$\mathbf{L} = \mathbf{U} \boldsymbol{\Lambda} \mathbf{U}^\top$$

with eigenvalues $0 = \lambda_1 \le \lambda_2 \le \cdots \le \lambda_n \le 2$.

## Graph Convolution via Polynomial Filters

A graph convolution with filter $g_\theta$ in the spectral domain is:

$$\mathbf{y} = g_\theta(\mathbf{L}) \mathbf{x} = \mathbf{U} g_\theta(\boldsymbol{\Lambda}) \mathbf{U}^\top \mathbf{x}$$

Direct spectral filtering is expensive ($O(n^2)$). ChebNet approximates $g_\theta$ with a Chebyshev polynomial expansion:

```python
# ChebNet convolution (simplified)
def chebnet_conv(x, L, K, theta):
    """
    x: node features [N, F]
    L: normalized Laplacian [N, N]
    K: polynomial order
    theta: learnable coefficients [K]
    """
    L_tilde = 2 * L / L.max() - torch.eye(L.shape[0])
    T = [x, L_tilde @ x]
    for k in range(2, K):
        T.append(2 * L_tilde @ T[-1] - T[-2])
    return sum(theta[k] * T[k] for k in range(K))
```

## Key Methods

| Method | Filter Type | Key Innovation |
|--------|-------------|----------------|
| ChebNet | Chebyshev polynomial | Localized spectral filters |
| GCN | 1st-order approx | Simplified propagation rule |
| GPR-GNN | Generalized PageRank | Signed polynomial coefficients |
| BernNet | Bernstein polynomial | Non-negative, bounded coefficients |
| APPNP | Power iteration | Personalized PageRank |

## Node-Adaptive Corrections

A key limitation of global polynomial filters is their inability to adapt to **local graph structure**. Nodes in dense regions of the graph exhibit fundamentally different spectral properties than nodes in sparse neighborhoods.

The node-adaptive eigenvalue correction idea proposes a per-node perturbation $\epsilon_v$ to the effective eigenvalue:

$$\tilde{\lambda}_v = \lambda + \epsilon_v(\text{struct}(v))$$

where $\text{struct}(v)$ encodes structural features like degree, local Dirichlet energy, and clustering coefficient. This perturbation is analytically equivalent to a transformation of polynomial coefficients via the binomial theorem:

$$g(\tilde{\lambda}_v) = \sum_{k=0}^K \theta_k (\lambda + \epsilon_v)^k = \sum_{k=0}^K \tilde{\theta}_k^{(v)} \lambda^k$$

This formulation retains pre-computability while enabling expressive per-node filtering.

## Conclusion

Spectral GNNs provide a principled, theoretically-grounded approach to graph learning. The transition from fixed to learnable, and from global to node-adaptive filters, represents an exciting ongoing research direction with substantial practical implications for heterogeneous and large-scale graphs.

---

*Keywords: ChebNet, BernNet, GPR-GNN, graph Laplacian, polynomial filter, eigenvalue, node-adaptive*
