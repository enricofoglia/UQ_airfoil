# UQ_airfoil
Comparison of uncertainty quantification (UQ) techniques for simple graph neural networks (GNNs). All models are trained to predict the aerodynamic performances of NACA four-digits airfoils

## Dataset and preprocessing
The dataset is composed of 500 NACA-XXXX airfoils discretized into 201 panels. The targets for the models are the $c_p$ distribution, computed with XFoil at angle of attack $\alpha=5^{\circ}$, and the lift to drag ratio $(c_L/c_D)$. In order to do so, the geometry is treated as an undirected graph where every node has two neighbors. As input, we use the components of the normalized tangent vectors and the length of the edge as edge features and position, curvature and Fourier modes (see `dataset.FourierEpicycles`) as node features (only the first 10 Fourier modes, without any additional data, seem to be sufficient to achive optimal performances).

## Base model
The encoder-process-decoder architecture as presented in (Battaglia2018) has been chosen as the base model. Each processing block uses a mini MLP as the non-linear processing function and a sum aggregation to pass between edge and node features. At the last layer a softmax aggregation is used to pass from node to global features in order to predict the final scalar (`model.EncodeProcessDecode`). 

## Epistemic uncertainty estimation
Since the targets are computed using a deterministic solver, the aleatoric uncertainty is considered to be zero. The epistemic part of the uncertainty is estimated via:
- The MC Dropout procedure as explained in (Gal2016) (`model.MCDroput`);
- Deep Ensemble as for the original paper by (Lak2017) (`model.Ensemble`);
- The reentrant neural network (ZigZag) methodology described in (Durasov2024) (`model.ZigZag`) 


## References
Battaglia, Peter W., et al. "Relational inductive biases, deep learning, and graph networks." arXiv preprint arXiv:1806.01261 (2018).

Gal, Yarin, and Zoubin Ghahramani. "Dropout as a bayesian approximation: Representing model uncertainty in deep learning." international conference on machine learning. PMLR, (2016).

Lakshminarayanan, Balaji, Alexander Pritzel, and Charles Blundell. "Simple and scalable predictive uncertainty estimation using deep ensembles." Advances in neural information processing systems 30 (2017).

Durasov, Nikita, et al. "Zigzag: Universal sampling-free uncertainty estimation through two-step inference." Transactions on Machine Learning Research (2024).
