#include <vector>
#include <cmath>
#include <tuple>

#include <algorithm>
#include <functional>
#include <array>
#include <iostream>
#include <iomanip>
#include <string_view>
#include <map>
#include <string>

#include <iostream>
#include <Eigen/Dense>

// autodiff include
#include <autodiff/forward/real.hpp>
#include <autodiff/forward/real/eigen.hpp>
#include <autodiff/forward/dual.hpp>
using namespace autodiff;

//
// To compile the code, the following libraries must be install (header files required only):
//
// .) Eigen: https://eigen.tuxfamily.org/
// .) autodiff: https://autodiff.github.io/
//
// Then, compile as follows, whereby the folders containtin the head files are specified:
//
// c++ filter.cpp -std=c++17 -I. -Ieigen-3.4.0 -Iautodiff
//

using Eigen::MatrixXd;
using namespace Eigen;



//
// Implement a template for a general non-linear discrete-time dynamic system:
//
// X[k+1] = f( X[k], U[k] )
// Y[k]   = g( X[k], U[k] ) 
//
// Herein, f is the transition function and g the output function.
//

template<typename T, auto N, auto N_IN, auto N_OUT, auto transition_function, auto output_function>
class NonlinearDynamicSystem {
    public:

    Matrix< double, N, N > A;
    Matrix< double, N, N_IN > B;
    Matrix< double, N_OUT, N > C;

    Vector< double, N > X;

    // template proxy
    static const int ORDER = N;
    static const int N_INPUTS = N_IN;
    static const int N_OUTPUTS = N_OUT;
    typedef T TYPE;


    NonlinearDynamicSystem( double c1, double c2, double c3 ) {
        reset();

        std::cout << "Created non-linear dynamic system of order " << ORDER << " with"
        << N_INPUTS << " input(s) and " << N_OUTPUTS << " output(s)." << std::endl;
    }

    void reset() {
        // X0
        int i;
        for (i = 0; i < N; ++i) {
            X[i] = 0.0;
        }
    }

    void set_states( const Vector< double, N > & X_set ) {
        X = X_set;
    }

    void compute_jacobians( 
        Matrix< double, N, N > &dFdX, 
        Matrix< double, N, N_IN > &dFdU,

        Matrix< double, N_OUT, N > &dYdX,
        Matrix< double, N_OUT, N_IN > &dYdU,

        const Vector< double, N_IN > & U 
    ) {
        // https://autodiff.github.io/tutorials/

        autodiff::VectorXreal X_lp(N);
        autodiff::VectorXreal U_lp(N_IN);

        // linearisation point
        X_lp = X;
        U_lp = U;

        VectorXreal F;

        dFdX = jacobian( transition_function, wrt(X_lp), at(X_lp, U_lp), F );
        dFdU = jacobian( transition_function, wrt(U_lp), at(X_lp, U_lp), F );

        // output function
        VectorXreal G;

        dYdX = jacobian( output_function, wrt(X_lp), at(X_lp, U_lp), G );
        dYdU = jacobian( output_function, wrt(U_lp), at(X_lp, U_lp), G );

    }

    void step( const Vector< double, N_IN > & U ) {

        auto X_next = transition_function( X, U );

        int i;
        for ( i = 0; i < N; ++i ) {
            X[i] = X_next[i][0];
        }

    }

    void output( const Vector< double, N_IN > & U, Vector< double, N_OUT > & Y ) {
        autodiff::VectorXreal X_lp(N);
        autodiff::VectorXreal U_lp(N_IN);

        // linearisation point
        X_lp = X;
        U_lp = U;

        autodiff::VectorXreal Y_ = output_function( X_lp , U_lp );

        int i;
        for (i = 0; i < N_OUT; ++i) {
            Y[i] = Y_[i][0];
        }
    }

    void print_states() const {
        int i;
        for (i = 0; i < N; ++i ) {
            std::cout << std::setw(14) << X[i];
        }

        std::cout << std::endl;
    }

};

//
// Implement a template for an extended Kalman filter (EKF).
// This implementation takes a model described by a non-linear
// dynamic system (SYSTEM) and uses it as a filter-internal model.
//
template <typename SYSTEM>
class Filter {
    public:

    SYSTEM system;

    Matrix< typename SYSTEM::TYPE, SYSTEM::ORDER, SYSTEM::ORDER > P;
    Matrix< typename SYSTEM::TYPE, SYSTEM::ORDER, SYSTEM::ORDER > Q;
    Matrix< typename SYSTEM::TYPE, SYSTEM::N_OUTPUTS, SYSTEM::N_OUTPUTS > R;

    Filter(
        SYSTEM & system_,
        Matrix < typename SYSTEM::TYPE, SYSTEM::ORDER, SYSTEM::ORDER > const & Q,
        Matrix < typename SYSTEM::TYPE, SYSTEM::N_OUTPUTS, SYSTEM::N_OUTPUTS > const & R
    ) 
      : system{system_}
      , Q(Q)
      , R(R)
    {
    }

    void reset( Vector < typename SYSTEM::TYPE, SYSTEM::ORDER > const & P_diagonal ) {

        P.diagonal() = P_diagonal;
        system.reset();
    }

    void step( 
        const Vector< typename SYSTEM::TYPE, SYSTEM::N_INPUTS > & U,
        const Vector< typename SYSTEM::TYPE, SYSTEM::N_OUTPUTS > & Y
    ) {
        // measurement Y
        auto z = Y; // output of system to observe

        Vector < typename SYSTEM::TYPE, SYSTEM::N_OUTPUTS > internal_model_output;
        system.output( U, internal_model_output );

        // estimation output residual vector
        auto e = z - internal_model_output;

        // step forward with the internal model
        system.step( U );
        auto X_apriori = system.X;

        // get jacobians
        Matrix< typename SYSTEM::TYPE, SYSTEM::ORDER, SYSTEM::ORDER > dFdX;
        Matrix< typename SYSTEM::TYPE, SYSTEM::ORDER, SYSTEM::N_INPUTS > dFdU;

        Matrix< typename SYSTEM::TYPE, SYSTEM::N_OUTPUTS, SYSTEM::ORDER > dYdX;
        Matrix< typename SYSTEM::TYPE, SYSTEM::N_OUTPUTS, SYSTEM::N_INPUTS > dYdU;

        system.compute_jacobians(dFdX, dFdU, dYdX, dYdU, U);
        
        // aliases
        auto F = dFdX;
        auto B = dFdU;
        auto H = dYdX;

        // predicted state covariance P(k|k-1)
        P = F * P * F.transpose() + Q;

        // Kalman gain
        auto S = H * P * H.transpose() + R;
        auto K = P * H.transpose() * S.inverse();

        // post priori state X(k|k)
        auto X_post = system.X + K * e;
        system.set_states(X_post);
        
        // post priori covariance
        auto I_44 = Matrix<typename SYSTEM::TYPE, SYSTEM::ORDER, SYSTEM::ORDER>::Identity();
        P = (I_44 - K * H) * P;
    }

    void print_states() const {
        system.print_states();
    }

};





//
// define concrete example
// Herein, a transition function f and an output function g is defined
// to descripe a pendulum system. Further, the system involves an actuator
// model with low pass characteristic.
//

autodiff::VectorXreal transition_function_test_1(
    const autodiff::VectorXreal & X,
    const autodiff::VectorXreal & u
) {
    autodiff::VectorXreal X_next(3);

    const double Ts = 0.1;    // sampling time [s]
    const double m = 1;       // mass [Kg]
    const double g = 9.81;    // gravitational constant
    const double c = 0.01;    // friction
    const double z_inf = 0.7; // pole of discrete-time low pass

    // actuation low pass characteristic
    X_next[0] = z_inf * X[0] + (1-z_inf) * u[0];

    // pendulum discretised using Euler forward 
    X_next[1] = X[1] + Ts * ( X[0] - m * g * sin( X[2] ) + c * X[1] ); // angular velocity
    X_next[2] = X[2] + Ts * ( X[1] ); // angle

    return X_next;
}


static autodiff::VectorXreal output_function_test_1(
    autodiff::VectorXreal & X,
    autodiff::VectorXreal & u
) {
    autodiff::VectorXreal Y(2);

    // define the system outputs
    Y[0] = X[0];
    Y[1] = X[2];

    return Y;
}


//
// Implement a simulation of a real system (ground truth) and
// an EKF that estimates the states of the former given the I/O
// signals.
//
void test_1()
{
    static const int N_IN  = 1;
    static const int N_OUT = 2;
    static const int N_ORDER = 3;
    static const int n_steps {30};
 
    typedef NonlinearDynamicSystem<
        double, 
        N_ORDER, N_IN, N_OUT, 
        transition_function_test_1, output_function_test_1
    > SYSTEM;


    Vector< double, N_IN > u;
    Vector< double, N_OUT > y;

    //
    SYSTEM nlds(0.1, 0.2, 0.3);
    SYSTEM nlds_internal_model(0.1, 0.2, 0.3);

    // disturb the initial states of the real system
    Vector <double, N_ORDER> X0;
    X0 << 0.0, 0.1, 0.0;

    nlds.set_states( X0 );

    // apply a constant input (unit step function)
    u[0] = 1.0;

    // covariances
    Matrix < double, N_ORDER, N_ORDER > Q;
    Q.diagonal() << 0.01, 0.01, 0.01;

    Matrix < double, N_OUT, N_OUT > R;
    R.diagonal() << 0.1, 0.1;

    // created Filter (EKF)
    Filter<SYSTEM> filter(nlds_internal_model, Q, R); 

    // set initial covariance of the estimate
    Vector< double, N_ORDER > P;
    P << 0.1, 0.1, 0.1;
    filter.reset( P );

    //
    // RUN
    //
    std::cout << "Run filter with system" << std::endl;

    //
    int i;

    // simulation loop
    nlds.print_states();
    for (i = 0; i < n_steps; ++i) {

        nlds.output( u, y );
        nlds.step( u );

        std::cout << "real X:    ";
        nlds.print_states();

        // feed the measurement y_ and the control input u to the filter
        filter.step( u, y );

        std::cout << "estimates for X:" << "                                                              ";
        filter.print_states();

    }
}


int main(int argc, char **argv)
{
    test_1();
}


