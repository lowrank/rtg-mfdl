# Topic 06 Project: Equivariant GNNs and Topological Analysis

## 1. Objective
Apply Geometric Deep Learning and Topological Data Analysis to solve complex structural problems. You will choose between modeling molecular systems with Equivariant GNNs or extracting topological signatures from point clouds.

## 2. Datasets

- **Option A (Geometry)**: **QM9** or **Kaggle: Molecular Property Prediction**.
  - [QM9 Link](https://paperswithcode.com/dataset/qm9)
  - [Kaggle Link](https://www.kaggle.com/c/champs-scalar-coupling)

- **Option B (Topology)**: **ModelNet10** (Point clouds of 3D objects).
  - [ModelNet Link](https://modelnet.cs.princeton.edu/)

---

## 3. Detailed Step-by-Step Implementation

### Option A: E(3)-Equivariant GNNs

1.  **Data Loading**: Use `torch_geometric` to load the QM9 dataset. Focus on predicting the **Dipole Moment** ($\mu$).
2.  **Architecture**: Implement an **EGNN (Equivariant Graph Neural Network)**.
- Message: $m_{ij} = \phi(h_i, h_j, \|x_i - x_j\|^2)$.
- Coordinate Update: $x_i = x_i + \sum (x_i - x_j) \psi(m_{ij})$.
- State Update: $h_i = h_i + \sum m_{ij}$.
3.  **Baseline**: Train a standard GCN that ignores the 3D coordinates $x$.
4.  **Verification**: After training, rotate a test molecule by 45 degrees. Verify that the scalar prediction remains identical, while any vector outputs (if any) rotate with the molecule.

### Option B: TDA for Shape Classification

1.  **Point Cloud Processing**: Sample 1024 points from each ModelNet10 object.
2.  **Persistence Summaries**:
- Use `Ripser` to compute the $H_1$ persistence diagram for each object.
- Convert diagrams into **Persistence Images** (fixed-size 2D grids).
3.  **Classification**:
- Train a CNN on the Persistence Images.
- Compare its accuracy to a **PointNet** baseline.
4.  **Analysis**: Identify which objects are "topologically distinct" (e.g., a chair with a hole in the back vs. a solid table).

---

## 4. Expected Results and Analysis

### Expected Observations

1.  **EGNN Advantage**: The EGNN should significantly outperform the GCN on QM9 because the 3D distances between atoms are crucial for chemical properties.
2.  **TDA Robustness**: TDA features should be almost perfectly invariant to the rotation and scaling of the point clouds, unlike raw coordinates.
3.  **Computational Cost**: TDA (specifically computing $H_1$) can be slow for large point clouds. You may need to use a "subsampling" strategy or the "Alpha Complex" instead of "Vietoris-Rips".

### Analysis Questions

- (Option A): How does the performance change if you only use distances $\|x_i - x_j\|$ but remove the coordinate update rule?
- (Option B): Can TDA distinguish between a "solid" sphere and a "hollow" sphere? What dimension of homology would you need?

---

## 5. Resources and Tips

- **GUDHI**: An excellent library for advanced topology (Alpha Complexes, Landscapes).
- **PyTorch Geometric (PyG)**: The industry standard for implementing GNNs.
- **E3NN**: A specialized library for Euclidean Equivariant Neural Networks using spherical harmonics.

