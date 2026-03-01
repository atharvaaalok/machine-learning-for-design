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
        - Data Generation:
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


- [Application of Convolutional Neural Network to Predict Airfoil Lift Coefficient](https://arxiv.org/abs/1712.10082) - 2017, Yao Zhang et al.
    <details>
    <summary>Main Takeaways</summary>

        - Motivation:
            - CNNs can prove to be better than MLPs at aerodynamic predictions by exploiting spatial
              coorelations.
            - Goal: Predict Cl.
        - Data Generation:
            - Xfoil as the solver at multiple AoA, Mach and Reynolds numbers.
            - UIUC airfoil database as the dataset.
            - AoA (-10, 30), Re (30,000 - 6,500,000), Mach (0.3, 0.8).
            - 40,000 simulations, augmented by adding inverted shape and Cl for a total 80,000.
        - Components:
            1. MLP baseline. Inputs: Re, Mach, AoA + 100 y-coords at pre-defined x-coords.
            2. AeroCNN-I. Inputs: 2x50 array of y-coords (upper and lower coords in each row).
              Perform convolution on this. Combine with Re, Mach, AoA in later FC layer.
            3. AeroCNN-II. Inputs: Artificial image (49x49) formed by the pixelated shape and
              coloring the pixels based on whether they are inside/on/outside and also based on the
              Mach number.
        - Key Ideas:
            - The "artificial image" synthesizes geometric (shape) and non-geometric boundaries
              (AoA, Mach) into a unified 2D array for CNNs to operate on.
        - Miscellaneous:
            - "Traditional way of solving problems in fluid mechanics is top-down and requires an
              understanding of the physics of the problem which is high-dimensional, multi-scale and
              nonlinear."
        - Potential Issue:
            - They claim that CNN can predict Cl values fast compared to solvers but they train
              using Xfoil data which is already fast. It is not clear what their approach has
              improved upon.
    </details>


- [Inverse Design of Airfoil using a Deep Convolutional Neural Network](https://arc.aiaa.org/doi/10.2514/1.J057894) - 2019, Vinothkumar Sekar et al.
    <details>
    <summary>Main Takeaways</summary>

        - Motivation:
            - CNN for gradient free airfoil design.
            - Goal: Given a Cp distribution, predict the airfoil shape.
        - Data Generation:
            - Xfoil as the solver at Re = 100,000, AoA = 3.
            - 1343 training, 143 testing airfoils.
        - Components:
            - Conv -> Pool -> FC layer architectures.
            - Input: Cp plots are converted to grayscale images. Passed either as two separate
              images of upper and lower Cp distribution (144x144x2, 216x216x2) or a combined image
              (144x144x1).
            - Output: (from FC layer) 70 y-coords (35 upper, 35 lower) at predefined x-coords.
        - Key Ideas:
            - Gradient free design of airfoils through a direct prediction of shape from Cp
              distribution.
        - Potential Issue:
            - They claim CNN helps reduce design time from hours (using solvers) to seconds but they
              train using Xfoil data which does not take hours for the optimization. It is not clear
              what their approach has improved upon.
    </details>


- [A Convolutional Neural Network Approach to Training Predictors for Airfoil Performance](https://arc.aiaa.org/doi/abs/10.2514/6.2017-3660) - 2017, Emre Yilmaz et al.
    <details>
    <summary>Main Takeaways</summary>

        - Motivation:
    </details>