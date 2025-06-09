extern crate nalgebra as na;
use std::error::Error;
use std::fmt;
use std::time::Instant;

use linfa::prelude::*;
use linfa_elasticnet::ElasticNet;
#[cfg(any(feature = "accelerate", feature = "openblas", feature = "netlib"))]
use nalgebra_lapack::Cholesky;
use ndarray::{Array, Array1};

/// Custom error type for Lasso and ADMM-related operations.
///
/// `LassoError` is the primary error type used throughout the library
/// for reporting failures in optimization algorithms, matrix operations,
/// and other computational tasks.
///
/// # Example
///
/// ```rust
/// # use srsadmm_core::utils::LassoError;
///
/// fn might_fail() -> Result<(), LassoError> {
///     Err(LassoError::from_string("Something went wrong".to_string()))
/// }
/// ```
#[derive(Debug)]
pub struct LassoError {
    /// The error message describing what went wrong
    message: String,
}

impl LassoError {
    /// Creates a new `LassoError` from a string message.
    ///
    /// # Arguments
    ///
    /// * `message` - A descriptive error message
    ///
    /// # Returns
    ///
    /// A new `LassoError` instance containing the provided message
    ///
    /// # Example
    ///
    /// ```rust
    /// # use srsadmm_core::utils::LassoError;
    ///
    /// let error = LassoError::from_string("Matrix inversion failed".to_string());
    /// ```
    pub fn from_string(message: String) -> Self {
        LassoError { message }
    }
}

impl fmt::Display for LassoError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl Error for LassoError {}

/// Computes the optimal objective value for Lasso regression using the linfa library.
///
/// This function solves the Lasso optimization problem using a third-party solver
/// to find the optimal objective value p*. It's primarily intended for testing
/// and validation purposes, not for production use.
///
/// # Arguments
///
/// * `a` - The feature matrix (m × n)
/// * `b` - The target vector (m × 1)  
/// * `lambda` - The L1 regularization parameter
///
/// # Returns
///
/// The optimal objective value p* = 0.5 * ||Ax* - b||² + λ||x*||₁
///
/// # Performance
///
/// This function prints timing information to help assess computational cost.
/// The objective is scaled by 1/m to match the average loss formulation.
///
/// # Example
///
/// ```rust,no_run
/// # use nalgebra as na;
/// # use srsadmm_core::utils::find_p_star_linfa;
///
/// let a = na::DMatrix::<f32>::new_random(100, 50);
/// let b = na::DMatrix::<f32>::new_random(100, 1);
/// let lambda = 0.1;
///
/// let p_star = find_p_star_linfa(&a, &b, lambda);
/// println!("Optimal objective value: {}", p_star);
/// ```
pub fn find_p_star_linfa(a: &na::DMatrix<f32>, b: &na::DMatrix<f32>, lambda: f32) -> f32 {
    println!("Finding p* using linfa");
    let start_time = Instant::now();
    let m = a.nrows();
    let n = a.ncols();

    let a_ndarray = Array::from_shape_fn((m, n), |(i, j)| a[(i, j)]);
    let b_ndarray = Array1::from_vec(b.as_slice().to_vec());

    let dataset = Dataset::new(a_ndarray.clone(), b_ndarray.clone());

    let model = ElasticNet::lasso()
        .penalty(lambda / m as f32) // Divide by m to properly scale w/ avg loss objective
        .with_intercept(false)
        .fit(&dataset)
        .expect("Failed to fit ElasticNet model");

    let x_star = model.hyperplane().to_owned();

    let ax_b = a_ndarray.dot(&x_star) - b_ndarray;
    let sq_norm = ax_b.mapv(|x| x.powi(2)).sum();
    let x_l1 = x_star.mapv(|x| x.abs()).sum();

    println!(
        "Fit model w/ linfa and calculated p* in {:?}",
        start_time.elapsed()
    );

    return 0.5 * sq_norm + lambda * x_l1;
}

/// Efficiently computes the inverse of (A^T A + ρI) for Lasso optimization.
///
/// This function implements an optimized strategy for computing matrix inverses
/// that arise in ADMM algorithms for Lasso regression. It automatically chooses
/// between two approaches based on the matrix dimensions:
///
/// 1. Sherman-Morrison-Woodbury formula for "fat" matrices (m < n/2)
/// 2. Cholesky decomposition for other cases
///    - Uses LAPACK's optimized Cholesky when any LAPACK backend feature is enabled
///      (accelerate, openblas, or netlib)
///    - Falls back to nalgebra's built-in Cholesky when no LAPACK backend is available
///
/// # Arguments
///
/// * `a` - The feature matrix (m × n)
/// * `rho` - The augmented Lagrangian penalty parameter
///
/// # Returns
///
/// The inverse matrix (A^T A + ρI)^(-1)
///
/// # Algorithm Selection
///
/// - If m < n/2: Uses Sherman-Morrison-Woodbury to avoid computing A^T A directly
/// - Otherwise: Uses Cholesky decomposition of A^T A + ρI
///
/// # Panics
///
/// Panics if the matrix is not invertible or Cholesky decomposition fails.
///
/// # Example
///
/// ```rust,no_run
/// # use nalgebra as na;
/// # use srsadmm_core::utils::fast_lasso_inverse;
///
/// let a = na::DMatrix::<f32>::new_random(100, 50);
/// let rho = 1.0;
///
/// let inverse = fast_lasso_inverse(&a, rho);
/// println!("Computed inverse with shape: {}×{}", inverse.nrows(), inverse.ncols());
/// ```
pub fn fast_lasso_inverse(a: &na::DMatrix<f32>, rho: f32) -> na::DMatrix<f32> {
    let m = a.nrows();
    let n = a.ncols();

    let eye_n = na::DMatrix::<f32>::identity(n, n);

    return if m < n / 2 {
        // For "fat" matrices, use Sherman-Morrison-Woodbury formula
        let rho_inv = 1.0 / rho;
        let eye_m = na::DMatrix::<f32>::identity(m, m);

        // I + A*A^T/rho
        let i_aat = a * a.transpose() * rho_inv + &eye_m;
        let i_aat_inv = i_aat
            .try_inverse()
            .ok_or(format!("Fast matrix inversion failed"))
            .unwrap();

        // (A^TA + rho*I)^-1 = rho^-1*I - rho^-2*A^T*(I+AA^T/rho)^-1*A
        eye_n * rho_inv - (a.transpose() * i_aat_inv * a) * (rho_inv * rho_inv)
    } else {
        // Cholesky decomposition (with LAPACK if available, fallback to nalgebra)
        let mut ata = a.transpose() * a;
        ata = ata + eye_n * rho;
        
        #[cfg(any(feature = "accelerate", feature = "openblas", feature = "netlib"))]
        {
            let l = Cholesky::new(ata).expect("Failed to compute Cholesky decomposition");
            l.inverse().expect("Failed to compute inverse")
        }
        
        #[cfg(not(any(feature = "accelerate", feature = "openblas", feature = "netlib")))]
        {
            // Fallback to nalgebra's built-in cholesky when LAPACK is not available
            let l = ata.cholesky().expect("Failed to compute Cholesky decomposition");
            l.inverse()
        }
    };
}

/// Computes factorization components for fused Lasso optimization.
///
/// This function precomputes matrix factorizations needed for efficient
/// solving of fused Lasso subproblems. It returns both the inverse matrix
/// and the right-hand side vector that can be reused across iterations.
///
/// # Arguments
///
/// * `a` - The feature matrix (m × n)
/// * `b` - The target vector (m × 1)
/// * `rho` - The augmented Lagrangian penalty parameter
///
/// # Returns
///
/// A tuple containing:
/// - The factorized inverse matrix for efficient system solving
/// - The precomputed right-hand side A^T b
///
/// # Algorithm Selection
///
/// Like `fast_lasso_inverse`, this function chooses between:
/// 1. Sherman-Morrison-Woodbury for fat matrices (m < n/2)
/// 2. Cholesky decomposition for other cases
///    - Uses LAPACK's optimized Cholesky when any LAPACK backend feature is enabled
///      (accelerate, openblas, or netlib)
///    - Falls back to nalgebra's built-in Cholesky when no LAPACK backend is available
///
/// # Example
///
/// ```rust,no_run
/// # use nalgebra as na;
/// # use srsadmm_core::utils::fused_lasso_factor;
///
/// let a = na::DMatrix::<f32>::new_random(100, 50);
/// let b = na::DMatrix::<f32>::new_random(100, 1);
/// let rho = 1.0;
///
/// let (factor, rhs) = fused_lasso_factor(&a, &b, rho);
/// // Use factor and rhs for efficient solving in ADMM iterations
/// ```
///
/// # Usage in ADMM
///
/// The returned components can be used to efficiently solve systems of the form:
/// (A^T A + ρI) x = A^T b + ρ(z - y)
///
/// by computing: x = factor * (rhs + ρ(z - y))
pub fn fused_lasso_factor(
    a: &na::DMatrix<f32>,
    b: &na::DMatrix<f32>,
    rho: f32,
) -> (na::DMatrix<f32>, na::DMatrix<f32>) {
    let m = a.nrows();
    let n = a.ncols();

    let eye_n = na::DMatrix::<f32>::identity(n, n);

    if m < n / 2 {
        let rho_inv = 1.0 / rho;
        let i_aat = a.transpose() * a * rho_inv + &eye_n;
        // I + A*A^T/rho
        let i_aat_inv = i_aat
            .try_inverse()
            .ok_or(format!("Fast matrix inversion failed"))
            .unwrap();

        // (A^TA + rho*I)^-1 = rho^-1*I - rho^-2*A^T*(I+AA^T/rho)^-1*A
        (
            eye_n * rho_inv - (a.transpose() * i_aat_inv * a) * (rho_inv * rho_inv),
            a.transpose() * b,
        )
    } else {
        let ata = a.transpose() * a + eye_n * rho;
        let atb = a.transpose() * b;

        // Cholesky decomposition (with LAPACK if available, fallback to nalgebra)
        #[cfg(any(feature = "accelerate", feature = "openblas", feature = "netlib"))]
        {
            let l = Cholesky::new(ata).expect("Failed to compute Cholesky decomposition");
            (l.inverse().expect("Failed to compute inverse"), atb)
        }
        
        #[cfg(not(any(feature = "accelerate", feature = "openblas", feature = "netlib")))]
        {
            // Fallback to nalgebra's built-in cholesky when LAPACK is not available
            let l = ata.cholesky().expect("Failed to compute Cholesky decomposition");
            (l.inverse(), atb)
        }
    }
}
