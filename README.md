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
            - Surrogate models require specific airfoil parameterizations. Neural Networks allow
              automatic feature extraction allowing generalization.
            - Goal: Predict Cp distribution at pre-defined x-coords for given airfoil.
        - Data Generation:
            - In-house panel code for inviscid, incompressible flow at 0 AoA.
            - UIUC database, 1562 airfoils, 1600 sample points on each.
        - Components:
            - CNN as main architecture. Also try 1) Pure Softmax classifier,
              2) Autoencoder + Softmax. CNNs outperform other approaches and form the main focus.
            - Treat the Cp prediction problem as classification by discretizing into bins.
            - Input: Array of (x, y) coordinate pairs of the entire airfoil.
            - Output: Cp value at each (x, y) point.
            - Trained using Cross-entropy loss.
        - Miscellaneous:
            - Errors are high at LE for both upper and lower surfaces due to high pressure
              variability.
    </details>


- [Prediction of Aerodynamic Flow Fields using Convolutional Neural Networks](https://link.springer.com/article/10.1007/s00466-019-01740-0) - 2019, Saakaar Bhatnagar and Yaser Afshar et al.
    <details>
    <summary>Main Takeaways</summary>

        - Motivation:
            - RANS solvers are time consuming, build a surrogate for speed up.
            - Goal: CNN to predict entire flow field (velocity and pressure) over 2D airfoils in
              turbulent flow at different Re, AoA.
        - Data Generation:
            - OVERTURNS as solver for compressible RANS.
            - Airfoils (S805, S809, S814). AoA (0, 20), Re (0.5, 1, 2, 3) * 1e6 for a total of 252
              simulations (3 * 21 * 4).
            - Structured C mesh with dimensions (384x124 = 48,856).
            - Mach = 0.2. Flow conditions are for wind turbines.
        - Components:
            - Encoder-Decoder CNN architecture similar to Guo et al.
            - Geometry: SDF on cartesian grid (150x150).
            - Loss: MSE + GS + L2. GS = Gradient Sharpening (they use central difference).
            - Validation: Mean Absolute Percentage Error (MAPE).
        - Key Ideas:
            - Their contribution: 1) Use RANS data, 2) Rigorous aerodynamic analysis, 3) Improve
              computational aspects of previous works (Guo et al. separated decoder -> shared
              decoder)
            - 10,000x speed up over RANS solver while maintaining <10% error.
            - Gradient sharpening improves error significantly and also improves the visual match.
            - Improved speed upon using shared decoder instead of separated decoder and 50% fewer
              parameters.
        - Miscellaneous:
            - Data-driven methods can augment (turbulence modeling) or replace (surrogates)
              high-fidelity solvers.
    </details>


- [Fast Flow Field Prediction over Airfoils using Deep Learning Approach](https://doi.org/10.1063/1.5094943) - 2019, Vinothkumar Sekar et al.
    <details>
    <summary>Main Takeaways</summary>

        - Motivation:
            - Solvers are time consuming, build a surrogate for speed up.
            - Previous CNN based surrogates are image-to-image and suffer from inaccuracies near the
              boundary because Cartesian image pixels cannot approximate curved airfoil boundaries.
            - Goal: Build a surrogate CNN + MLP that can predict flow field at arbitrary (x, y).
        - Data Generation:
            - CNN (geometry): 216x216 images of 1550 airfoils from UIUC database. Trained like an
              autoencoder reconstruct airfoil y-coords (35 upper, 35 lower).
            - MLP (flow): Open-FOAM as solver. 110 NACA airfoils, AoA (0, 14), Re (100, 2000).
              Randomly choose 6 AoA, 8 Re for each airfoil. Total 110 * 48 = 5280 simulations.
            - Incompressible, laminar, steady flow over airfoils.
        - Components:
            - 2 step process: 1) CNN, 2) MLP.
            - CNN: Extract the geometry features. Trained like an autoencoder: image of airfoil ->
              feature vector P1-P16 -> reconstruct airfoil y-coords (35 upper, 35 lower).
            - MLP: Use the extracted features with flow conditions (Re, AoA) and (x, y) coordinates
              to predict flow at arbitrary boundary points.
        - Key Ideas:
            - **Improved accuracy** by predicting flow at arbitrary (x, y) compared to
              image-to-image CNNs. Often predictions near the boundary are more important.
            - **Computationally efficient** compared to CFD solvers which due to BC at farfield
              require large meshes. Here we can do inference only on the boundary as well.
            - This approach can also work for optimization. Optimize over the feature vector P1-P16
              and then use decoder to reconstruct airfoil y-coords.
    </details>


- [Fast Pressure Distribution Prediction of Airfoils using Deep Learning](https://doi.org/10.1016/j.ast.2020.105949) - 2020, Xinyu Hui et al.
    <details>
    <summary>Main Takeaways</summary>

        - Motivation:
            - Solvers are time consuming, build a surrogate for speed up.
            - Goal: CNN to predict Cp distribution in transonic flow with shock waves.
        - Data Generation:
            - Base RAE2822 airfoil deformed using Free-Form Deformation (FFD) and Latin Hypercube
              Sampling (LHS) to create 1500 airfoils.
            - CFL3D as solver.
            - Mach = 0.734, Re = 6.5 * 1e6, AoA = 2.79.
            - Cp is found at pre-decided x-coords using linear interpolation.
            - Robustness of model validated by repeating everything for S809 airfoil at subsonic
              conditions.
        - Components:
            - CNN that takes as input SDF images and outputs Cp at y-coords at 49 pre-decided
              x-coords. 2 networks, one for each Cp upper and Cp lower.
            - Input: 32x32 images with SDF for geometry.
            - Cp is discretized into bins and they perform classification instead of regression.
        - Key Ideas:
            - Model is able to capture unseen phenomenon like double shocks and strong shocks.
            - 500x speed up compared to their solver.
    </details>


- [Lat-Net - Compressing Lattice Boltzmann Flow Simulations using Deep Neural Networks](https://arxiv.org/abs/1705.09036) - 2017, Oliver Hennigh
    <details>
    <summary>Main Takeaways</summary>

        - Motivation:
            - Fluid simulations are computationally and memory demanding.
            - Surrogate for time-dependent turbulence because that is hard and needs high spatial
              and temporal resolution.
            - Same approach works for other PDEs solved by LBM, they do electromagnetism too.
            - Goal: CNN based autoencoder approximation to reduce compute time and memory of LBM.
            - Contributions: 1) Reduce memory requirements especially in 3D where it grows cubically
              with grid size, 2) Train on small simulation, generalize on large, 3) Method
              applicable to many physical simulations.
        - Data Generation:
            - MechSys library for simulations.
            - 2D - Train: 50 simulations 256x256, D2Q9 with 8 objects of random size and position.
              Test: 256x256, 1024x1024 with similar objects, 256x512 with vehicle cross-sections.
            - 3D - Train: 50 simulations 40x40x160, D3Q9 with 4 sphere of fixed size but random
              position. Test: 40x40x160, 160x160x160.
            - Electromagnetism data - Train: 256x256, Test: 512x512.
        - Components:
            - Lat-Net has 3 components: 1) Encoder - encode state and BC, 2) Map encoded state to
              next state in time, 3) Decode actual state from codes.
            - Inputs: 1) ft - flow state at t (nx, ny, 9) for 2D or (nx, ny, nz, 15) for 3D and
              2) b - boundary (nx, ny, nz, 1) with 0/1 values for occupancy.
            - Networks compress ft -> gt and b -> bmul, badd (same size as gt).
            - In LBM, BC is used at every time-step so they also add BC to state gt -> gt * bmul +
              badd. This allowed BC info to be rooted throughout simulation.
            - Modified state is simulated using another network gt -> gt+1. Each network step
              corresponds to 60/120 solver steps.
            - Each gt is decoded using a decoder network gt -> ft.
            - Networks are fully convolutional allowing training on small simulations and testing on
              large ones.
            - Loss: MSE + GDL (Gradient Difference Loss).
        - Key Ideas:
            - Using fully convolutional network allows predicting flow for larger simulations but
              the predictions are unstable.
            - The dynamic state update is fast (9x) but decoding is slow and overall no gains.
            - Time for computing flow at a point/line/plane is much cheaper thanks to fully
              convolutional layers.
        - Miscellaneous:
            - LBM can be seen through the lens of convolution. Streaming step is like 3x3 conv and
              collision step is like 1x1 conv. So their CNN approach compresses a large CNN into a
              small one.
    </details>


- [A Deep Learning Approach for the Transonic Flow Field Predictions Around Airfoils](https://doi.org/10.1016/j.compfluid.2022.105312) - 2022, Cihat Duru et al.
    <details>
    <summary>Main Takeaways</summary>

        - Motivation:
            - Goal: CNN surrogate to predict flow field in hard flow conditions.
            - Consider steady, compressible flow at high AoA, M = 0.7 and Re = 6 * 1e6. This makes
              the test case (shock, stall) much harder than incompressible and transonic cases that
              have been studied before.
            - Extension of CNNFOIL work.
        - Data Generation:
            - Subset of UIUC airfoils. 204 airfoils, AoA (-10, 20), M = 0.7, Re = 6 * 1e6. Total
              simulations 204 * 31 = 6324.
            - In-house CFD solver, RANS.
            - Structured O-grid meshes with yplus < 1 and far-field at 500x chord.
            - Field values are interpolated to the cartesian grid used for SDF input image.
        - Components:
            - Input: Geometry as SDF images 256x256 modified to be 0 within airfoil.
            - Output: Pressure and Mach fields.
            - CNN maps SDF image to Pressure and Mach fields.
            - Loss: MSE. Evaluation: Absolute error, MAPE, AWT (Accuracy with threshold) =
              max(|y/y'|, |y'/y|) < delta.
        - Key Ideas:
            - Shock location predicted within 2% error. 
            - Cp and M errors are large and localized at shock location. Model captures overall flow
              structure very well.
            - OOD generalization test on NACA0012. Performance similar to that on training data.
            - They argue that ML model is for low fidelity, hence unfair to compare performance with
              high accuracy CFD. They also run less accurate CFD simulation that gives a 20x speed
              boost over high accuracy CFD. They find that it is less accurate than model while
              still being 100x slower. Therefore models perform better than plain less accuracy CFD.
        - Miscellaneous:
            - Treat airfoil rotated by AoA as a new shape.
            - Cp errors larger than M as Cp field has larger gradients across shocks.
            - Only pressure distribution is available through CNN prediction. No shear stress. That
              will require much finer resolution than 256x256. So only pressure lift and drag.
            - Good match in Cd prediction at large AoA as most Cd comes from pressure. At small AoA,
              viscous forces have significant contribution and we see large errors.
            - There is a loss of accuracy in the flow field at the flow interpolation step itself.
              So model is actually trained on data with errors. Eg: slight shock shift.
        - Potential Issues:
            - Data points with > 100% error are excluded from MAPE calculations. 2% of data are
              outliers.
    </details>


- [CNNFOIL - Convolutional Encoder Decoder Modeling for Pressure Fields Around Airfoils](https://doi.org/10.1007/s00521-020-05461-x) - 2020, Cihat Duru et al.
    <details>
    <summary>Main Takeaways</summary>

        - Motivation:
    </details>