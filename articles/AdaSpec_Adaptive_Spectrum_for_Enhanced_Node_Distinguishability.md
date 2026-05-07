## Information
1. Authors: Fangbing Liu & Qing Wang, The Australian National University
2. 2026 ICLR
3. [https://github.com/Mia-321/AdaSpec](https://github.com/Mia-321/AdaSpec)

## Abstract
1. Spectral GNNs 中的**node distinguishability** 不够透明
    - 本文探索图矩阵和节点特征如何一起影响节点的可区分度
    - 可区分的节点数目的理论下界由两个因素一起决定：不同的特征值和节点特征的非零频率部件
2. AdaSpec：
    - 不增加计算复杂度的情况下的自适应图矩阵生成模块
    - 证明其可以保留置换不变性

## 1 Introduction
1. Node distinguishability： GNN 将拓扑/特征上不同的节点映射到不同的 embedding 的能力
2. AdaSpec：
    - plug-in
    - 保留置换不变性（$\textcolor{red}{为什么需要这个性质，主要由于理论上不完备吗}$）
    - 保证图的连通性：学习到的表征能反映 the underlying graph structure

## 3 Preliminaries
1. Structurally equivalent: 两个节点拥有完全相同的邻居，$s_u\sim s_v$，交换这两个节点不会引起图邻接关系发生变化
2. $G$ 的所有自同构映射构成自同构群 $Aut(G)$
3. **Definition 3.1 Permutation Equivalence：** $f(\pi(G))=\pi(f(G))$ 

## 4 Node Distinguishability of Spectral GNNs
1. **Definition 4.1 Node Distinguishability：** $f(G)_v \neq f(G)_u \text{for all } v,u\in G, v\nsim u,$ 即只有两个节点不是同构点，则表征不相同
2. **Definition 4.2 Spectrum and Frequency Components：** $M=U\Lambda U^T\in \mathbb R^{n\times n}$，$M$ 的 spectrum 为特征值的 multiset： $spec(M)=\{\{\lambda_1,...,\lambda_n\}\}， \lambda_i=\Lambda_{ii}$