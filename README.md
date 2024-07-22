# UQ_airfoil
Comparison of uncertainty quantification (UQ) techniques for simple graph neural networks (GNNs). All models are trained to predict the aerodynamic performances of NACA four-digits airfoils

## Base model
The dataset is composed of 500 NACAXXXX airfoils discretized into 200 panels. The targets for the models are the $c_p$ distribution, computed with XFoil at angle of attack $\alpha=5^{\circ}$, and the lift to drag ratio $(c_L/c_D)$. In order to do so, the geometry is treated as an undirected graph where every node has two neighbors. For the modeling we chose to go with the encoder-process-decoder architecture as presented in (Battaglia2018).



## References
Battaglia, Peter W., et al. "Relational inductive biases, deep learning, and graph networks." arXiv preprint arXiv:1806.01261 (2018).
