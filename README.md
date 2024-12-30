# POI-ESD: Points of Interest-Based Edge Service Deployment

This repository serves as a comprehensive resource accompanying the research paper "POI Distribution-Based and Topology-Aware Edge Service Deployment Optimization." It provides datasets, algorithms, and implementations to facilitate replication of our experiments and further exploration in edge service deployment.

### Contributions:

- **Integrated Dataset**: The datasets combine real-world edge server data from the [Shanghai Telecom dataset](http://sguangwang.com/TelecomDataset.html) with extensive POI data obtained from the [Baidu Maps API](https://lbsyun.baidu.com/). This fusion enables an in-depth analysis of geographical demand distributions and their impact on service deployment strategies.
- **Novel Algorithms**: The repository includes state-of-the-art methods and our proposed Multi-objective Topology-based Genetic Algorithm (MTGA), which optimizes service deployment by balancing geographical POI relevance and topological network coverage.
- **Open Resources**: Researchers and practitioners can freely access, experiment with, and extend the provided datasets and algorithms.

### How to Use:

- **Datasets**: To use the datasets or adapt them to your research, refer to the descriptions under the "Datasets" section below. The data includes detailed edge server configurations, network topology, POI attributes, and precomputed relevance scores.
- **Algorithms**: For experimental replication or adaptation, refer to the implementation details and code under the "Algorithms" section. This includes baseline methods, comparison approaches, and our proposed MTGA.

<!-- ### Citing This Work: -->

<!-- If you find our datasets or methods useful in your research, please cite our paper as follows: -->

<!-- ```latex
@article{your_citation_key, title={POI Distribution-Based and Topology-Aware Edge Service Deployment Optimization}, author={Your Authors}, journal={Your Journal}, year={2024} }
``` -->

<!-- We hope this repository supports and inspires further advancements in edge service deployment research.

--- -->

## Datasets

The datasets in this repository are derived from the integration of real-world data from the [Shanghai Telecom dataset](http://sguangwang.com/TelecomDataset.html) and POI information obtained via the [Baidu Maps API](https://lbsyun.baidu.com/). These datasets encompass edge servers, edge network topology, and Points of Interest (POIs) from three distinct urban regions: SHH Telcom@1, SHH Telcom@2, and SHH Telcom@3.

### Experimental DataSets

- **SHH Telcom@1**: A compact urban region (Wujiaochang) with diverse functionalities such as educational institutions and shopping centers.
- **SHH Telcom@2**: A medium-scale urban area (Huaihai Park) blending residential neighborhoods and business districts.
- **SHH Telcom@3**: A large-scale, high-density commercial hub (Lujiazui) with significant financial, tourist, and corporate activities.

The following table summarizes the key attributes of the three datasets:

| Parameter                   | SHH Telcom@1 | SHH Telcom@2 | SHH Telcom@3 |
| --------------------------- | ------------ | ------------ | ------------ |
| **Location**                | Wujiaochang  | Huaihai Park | Lujiazui     |
| **Geographical Range**      | 2000×1500 m² | 3000×2500 m² | 5000×4000 m² |
| **Number of Servers**       | 27           | 51           | 85           |
| **Number of POIs**          | 182          | 321          | 483          |
| **Average Coverage Radius** | 220 m        | 230 m        | 260 m        |

### Dataset Files

Each dataset folder includes three primary files: `poi.csv`, `edge.csv`, and `server.csv`. The following table provides a brief description of each file:

| Dataset File | Description                                                                                                                   |
| ------------ | ----------------------------------------------------------------------------------------------------------------------------- |
| `poi.csv`    | `id, name, area, type, x, y, rel` - Represents the POI ID, name, geographical area, type, x and y coordinates, and relevance. |
| `edge.csv`   | `s1, s2, distance` - Represents the IDs of two connected servers and the distance between them.                               |
| `server.csv` | `id, x, y, radius, budget` - Represents the server ID, coordinates, coverage radius, and deployment cost.                     |

---

## Algorithms

### 1. **Baseline and Comparison Methods**

The following algorithms have been implemented and can be found in the `algorithm/` directory:

- **Random**: Randomly deploys services across edge servers under budget constraints.
- **BEAD**: Maximizes distinct POI coverage under budget constraints.  
   Reference: [F. Chen, J. Zhou, X. Xia, H. Jin, and Q. He, “Optimal Application
  Deployment in Mobile Edge Computing Environment,” in 2020 IEEE
  13th International Conference on Cloud Computing (CLOUD), 2020,
  pp. 184–192.](https://doi.org/10.1109/TMC.2020.2970698)
- **RBEAD**: Ensures redundant POI coverage to improve reliability.  
   Reference: [B. Li, Q. He, G. Cui, X. Xia, F. Chen, H. Jin, and Y. Yang, “READ:
  Robustness-Oriented Edge Application Deployment in Edge Computing
  Environment,” IEEE Transactions on Services Computing, vol. 15, no. 3,
  pp. 1746–1759, 2022.](https://ieeexplore.ieee.org/document/9163305)
- **CRBEAD**: [Jointly optimizes POI coverage and redundancy.  
   Reference: F. Chen, J. Zhou, X. Xia, Y. Xiang, X. Tao, and Q. He, “Joint
  Optimization of Coverage and Reliability for Application Placement in
  Mobile Edge Computing,” IEEE Transactions on Services Computing,
  vol. 16, no. 6, pp. 3946–3957, 2023.](https://ieeexplore.ieee.org/document/10196050)

### 2. **Proposed Method**

#### MTGA (Multi-objective Topology-based Genetic Algorithm)

The proposed method, MTGA, is a multi-objective topology-based genetic algorithm that optimizes service deployment by balancing geographical POI relevance and topological extension. It encodes deployment strategies as graph-based chromosomes and employs relevance-aware matching and decay-aware heuristics during genetic operations. MTGA incorporates balanced fitness evaluation for multi-objective optimization and offers two variants: `MTGA-M` and `MTGA-T`.

- **Objective**: Optimizes service deployment by balancing geographical POI relevance and topological extension.
- **Key Features**:
  - Encodes deployment strategies as graph-based chromosomes.
  - Employs relevance-aware matching and decay-aware heuristics during genetic operations.
  - Incorporates balanced fitness evaluation for multi-objective optimization.
- **Variants**:
  - `MTGA-M`: Focuses on maximizing relevance-aware matching.
  - `MTGA-T`: Focuses on maximizing topological coverage with decay consideration.

---
