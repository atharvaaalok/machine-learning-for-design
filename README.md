# Machine Learning for Design
A curated list of resources on machine learning for fluid flow, structures and design optimization.


## Machine Learning for Fluid Flow

### Papers
- [Convolutional Neural Networks for Steady Flow Approximation](https://dl.acm.org/doi/pdf/10.1145/2939672.2939738) - 2016, Xiaoxiao Guo et al.
    <details>
    <summary>Main Takeaways</summary>

        - Motivation:
            - CFD is: 1) compute intensive, 2) memory demanding and 3) time-consuming.
            - Want real-time feedback for early stage design. CFD analysis is typically only for
              final stage design. In early design high fidelity not needed.
        - Data generation:
            - LBM solver. Handles complex geometry and trivially parallelizable.
            - Non-uniform steady laminar flow in 2D and 3D.
            - Re = 20.
            - Training dataset - primitive shapes like triangles, quadrilaterals etc.
            - Testing dataset - 2D car dataset.
            - 3D dataset of sphere-prism pairs for multiple boundaries.
        - Components:
            1. SDF pixel/voxel for geometry.
            2. CNN encoder.
            3. CNN deconvolution decoder.
        - Key Ideas:
            - SDF provides both local and global geometry details. Works much better than binary
              occupancy representation.
            - 300x faster than GPU and 12,000x faster than CPU based LBM solvers.
            - Train on - MSE. Eval metric - Average relative error.
        - Miscellaneous:
            - In large mesh based CFD data the SDF can have a wide range of values and it might be
              helpful to normalize that to [-1, 1].
            - "Traditional CFD simulation suffers from long response time, predominantly because of
              the complexity of the underlying physics and the historical focus on accuracy."
    </details>