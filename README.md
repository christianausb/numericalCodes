# numericalCodes
Collection of codes for numeric computations

## Extended Kalman filtering in c++

This code uses Eigen and the automatic differentiation to implement an extended Kalman filter for a given nonlinear dynamic system. Herein, the Jacobian matrix needed to compute the linearization of the nonlinear dynamics is computed using automatic differentiation. Hence, a manual (analytic) derivation and implementation is not needed. The implementation is based on templates; therefore, it is convenient to customize the number of states, outputs, and inputs.
