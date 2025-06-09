use std::marker::PhantomData;
use std::future::Future;
use std::sync::Arc;
use tokio::sync::{Mutex, RwLock};

use crate::{
    resource::StorageConfig,
    timing::{TimingTracker, time_async_fn},
    utils::LassoError,
};

/// ADMM execution context containing shared global and local state.
///
/// The `ADMMContext` encapsulates the shared state and configuration needed
/// for ADMM problem solving. It provides thread-safe access to both global
/// state (shared across all subproblems) and local state (specific to individual
/// subproblems) through appropriate synchronization primitives.
///
/// # Type Parameters
///
/// * `G` - The type of the global state shared across all subproblems
/// * `S` - The type of the local state specific to each subproblem
///
/// # Synchronization
///
/// - Global state uses `RwLock` allowing multiple concurrent readers or one writer
/// - Local state uses `Mutex` for exclusive access during modifications
///
/// # Example
///
/// ```rust,no_run
/// use srsadmm_core::problem::{ADMMContext, ADMMConfig};
/// use srsadmm_core::resource::StorageConfig;
/// use std::sync::Arc;
/// use tokio::sync::{Mutex, RwLock};
///
/// struct GlobalState {
///     iteration: usize,
/// }
///
/// struct LocalState {
///     subproblem_id: usize,
/// }
///
/// # fn example() {
/// let global = Arc::new(RwLock::new(GlobalState { iteration: 0 }));
/// let local = Arc::new(Mutex::new(LocalState { subproblem_id: 0 }));
/// let config = ADMMConfig {
///     storage: StorageConfig::new(/* ... */),
/// };
///
/// let context = ADMMContext {
///     config,
///     global,
///     local,
/// };
/// # }
/// ```
pub struct ADMMContext<G, S> {
    /// Configuration settings for the ADMM solver
    pub config: ADMMConfig,
    /// Global state shared across all subproblems with read-write lock
    pub global: Arc<RwLock<G>>,
    /// Local state specific to subproblems with mutex protection
    pub local: Arc<Mutex<S>>,
}

/// Configuration settings for ADMM execution.
///
/// `ADMMConfig` contains configuration parameters that control various
/// aspects of the ADMM algorithm execution, including storage backend
/// settings and optimization parameters.
///
/// # Fields
///
/// * `storage` - Configuration for data storage backends (local, S3, memory)
///
/// # Example
///
/// ```rust,no_run
/// # use srsadmm_core::problem::ADMMConfig;
/// # use srsadmm_core::resource::{StorageConfig, LocalConfig, S3Config};
/// # use std::path::Path;
///
/// let local_config = LocalConfig::new(Path::new("/tmp"), "admm_");
/// let s3_config = S3Config::new("my-bucket", "us-west-2", "admm/");
/// let storage_config = StorageConfig::new(local_config, s3_config);
///
/// let config = ADMMConfig {
///     storage: storage_config,
/// };
/// ```
#[derive(Clone)]
pub struct ADMMConfig {
    /// Storage backend configuration for variables and intermediate results
    pub storage: StorageConfig,
}

/// Trait defining the complete interface for ADMM problems.
///
/// This trait combines access to the problem context with asynchronous methods
/// that implement the core ADMM algorithm steps. Each method corresponds to a
/// specific phase of the ADMM iteration process.
///
/// # Type Parameters
///
/// * `G` - The type of the global state shared across all subproblems
/// * `S` - The type of the local state specific to each subproblem
///
/// # ADMM Algorithm Steps
///
/// The methods in this trait implement the standard ADMM algorithm:
/// 1. `context` - Provide access to the problem context
/// 2. `precompute` - Perform any necessary one-time computations
/// 3. `update_x` - Update the primal variable x
/// 4. `update_z` - Update the auxiliary variable z
/// 5. `update_y` - Update the dual variable y
/// 6. `update_residuals` - Compute primal and dual residuals
/// 7. `check_stopping_criteria` - Check if the algorithm should terminate
pub trait ADMMProblem<G, S> {
    /// Returns a reference to the ADMM context containing global and local state.
    fn context(&self) -> &ADMMContext<G, S>;

    /// Performs one-time precomputation before the ADMM iterations begin.
    ///
    /// This method is called once before the main ADMM loop starts and can be used
    /// to perform expensive computations that only need to be done once, such as
    /// matrix factorizations or data preprocessing.
    ///
    /// # Returns
    ///
    /// * `Ok(())` if precomputation succeeds
    /// * `Err(LassoError)` if an error occurs during precomputation
    ///
    fn precompute(&self) -> impl Future<Output = Result<(), LassoError>>;

    /// Updates the primal variable x in the ADMM algorithm.
    ///
    /// This corresponds to the x-update step in ADMM, typically involving
    /// solving a subproblem to minimize the augmented Lagrangian with respect to x.
    ///
    /// # Returns
    ///
    /// * `Ok(())` if the x-update succeeds
    /// * `Err(LassoError)` if an error occurs during the update
    fn update_x(&self) -> impl Future<Output = Result<(), LassoError>>;

    /// Updates the auxiliary variable z in the ADMM algorithm.
    ///
    /// This corresponds to the z-update step in ADMM, often involving
    /// a proximal operator or projection to enforce constraints.
    ///
    /// # Returns
    ///
    /// * `Ok(())` if the z-update succeeds
    /// * `Err(LassoError)` if an error occurs during the update
    fn update_z(&self) -> impl Future<Output = Result<(), LassoError>>;

    /// Updates the dual variable y (Lagrange multipliers) in the ADMM algorithm.
    ///
    /// This corresponds to the y-update step in ADMM, typically a simple
    /// gradient ascent step on the dual variable.
    ///
    /// # Returns
    ///
    /// * `Ok(())` if the y-update succeeds
    /// * `Err(LassoError)` if an error occurs during the update
    fn update_y(&self) -> impl Future<Output = Result<(), LassoError>>;

    /// Computes the primal and dual residuals for convergence checking.
    ///
    /// This method calculates the residuals used to assess convergence of the
    /// ADMM algorithm, including the primal residual ||Ax + Bz - c|| and
    /// the dual residual ||A^T y||.
    ///
    /// # Returns
    ///
    /// * `Ok(())` if residual computation succeeds
    /// * `Err(LassoError)` if an error occurs during computation
    fn update_residuals(&self) -> impl Future<Output = Result<(), LassoError>>;

    /// Checks whether the ADMM algorithm should terminate.
    ///
    /// This method evaluates stopping criteria such as convergence tolerances
    /// on the primal and dual residuals, or other problem-specific conditions.
    ///
    /// # Returns
    ///
    /// * `Ok(true)` if the algorithm should stop (converged)
    /// * `Ok(false)` if iterations should continue
    /// * `Err(LassoError)` if an error occurs during evaluation
    fn check_stopping_criteria(&self) -> impl Future<Output = Result<bool, LassoError>>;
}

/// ADMM solver that orchestrates the iterative optimization process.
///
/// The `ADMMSolver` manages the execution of the ADMM algorithm, including
/// timing tracking, iteration control, and result export capabilities.
/// It works with any problem that implements the `ADMMProblem` trait.
///
/// # Type Parameters
///
/// * `G` - The type of the global state shared across all subproblems
/// * `S` - The type of the local state specific to each subproblem  
/// * `P` - The problem type that implements `ADMMProblem<G, S>`
///
/// # Example
///
/// ```rust,no_run
/// # use srsadmm_core::problem::{ADMMSolver, ADMMProblem};
/// # async fn example<P: ADMMProblem<(), ()>>(problem: P) -> Result<(), Box<dyn std::error::Error>> {
/// let mut solver = ADMMSolver::new(problem, 1000);
/// solver.solve().await?;
/// solver.print_timing_summary();
/// solver.export_all_timings("results")?;
/// # Ok(())
/// # }
/// ```
pub struct ADMMSolver<G, S, P>
where
    P: ADMMProblem<G, S>,
{
    /// The problem instance implementing the ADMM algorithm steps
    problem: P,
    /// Maximum number of iterations before termination
    max_iter: usize,
    /// Tracks timing information for performance analysis
    timing_tracker: TimingTracker,
    /// Phantom data for global state type
    _global: PhantomData<G>,
    /// Phantom data for subproblem state type
    _subproblem: PhantomData<S>,
}

impl<G, S, P> ADMMSolver<G, S, P>
where
    P: ADMMProblem<G, S>,
{
    /// Creates a new ADMM solver with the specified problem and iteration limit.
    ///
    /// # Arguments
    ///
    /// * `problem` - The problem instance implementing `ADMMProblem`
    /// * `max_iter` - Maximum number of iterations before termination
    ///
    /// # Returns
    ///
    /// A new `ADMMSolver` instance ready to solve the problem
    pub fn new(problem: P, max_iter: usize) -> Self {
        ADMMSolver {
            problem,
            max_iter,
            timing_tracker: TimingTracker::new(),
            _global: PhantomData,
            _subproblem: PhantomData,
        }
    }

    /// Solves the ADMM problem by iterating until convergence or max iterations.
    ///
    /// This method executes the main ADMM algorithm loop, calling the appropriate
    /// update methods in sequence and tracking timing information for each step.
    /// The algorithm terminates when either the stopping criteria are met or
    /// the maximum number of iterations is reached.
    ///
    /// # Returns
    ///
    /// * `Ok(())` if the algorithm completes successfully
    /// * `Err(LassoError)` if an error occurs during any step
    ///
    /// # Algorithm Steps
    ///
    /// 1. Precompute any necessary values
    /// 2. For each iteration:
    ///    - Update x (primal variable)
    ///    - Update z (auxiliary variable)  
    ///    - Update y (dual variable)
    ///    - Compute residuals
    ///    - Check stopping criteria
    ///    - Continue if not converged and under max iterations
    pub async fn solve(&mut self) -> Result<(), LassoError> {
        time_async_fn(
            &mut self.timing_tracker,
            "precompute",
            self.problem.precompute(),
        )
        .await?;
        let mut i = 0;
        loop {
            self.timing_tracker.start_iteration();

            println!("[ADMMSolver] ===== Iteration: {} =====", i);

            time_async_fn(
                &mut self.timing_tracker,
                "update_x",
                self.problem.update_x(),
            )
            .await?;
            time_async_fn(
                &mut self.timing_tracker,
                "update_z",
                self.problem.update_z(),
            )
            .await?;
            time_async_fn(
                &mut self.timing_tracker,
                "update_y",
                self.problem.update_y(),
            )
            .await?;
            time_async_fn(
                &mut self.timing_tracker,
                "update_residuals",
                self.problem.update_residuals(),
            )
            .await?;

            let should_stop = time_async_fn(
                &mut self.timing_tracker,
                "check_stopping_criteria",
                self.problem.check_stopping_criteria(),
            )
            .await?;
            if should_stop {
                break;
            }
            i += 1;
            if i == self.max_iter {
                break;
            }
        }
        Ok(())
    }

    /// Exports step timing data to a CSV file for analysis.
    ///
    /// # Arguments
    ///
    /// * `filename` - Path to the output CSV file
    ///
    /// # Returns
    ///
    /// * `Ok(())` if export succeeds
    /// * `Err(LassoError)` if file writing fails
    pub fn export_step_timings(&self, filename: &str) -> Result<(), LassoError> {
        self.timing_tracker.write_step_timings_to_csv(filename)
    }

    /// Exports lambda timing data to a CSV file for analysis.
    ///
    /// # Arguments
    ///
    /// * `filename` - Path to the output CSV file
    ///
    /// # Returns
    ///
    /// * `Ok(())` if export succeeds
    /// * `Err(LassoError)` if file writing fails
    pub fn export_lambda_timings(&self, filename: &str) -> Result<(), LassoError> {
        self.timing_tracker.write_lambda_timings_to_csv(filename)
    }

    /// Gets mutable access to the timing tracker for custom timing operations.
    ///
    /// # Returns
    ///
    /// A mutable reference to the internal `TimingTracker`
    pub fn timing_tracker_mut(&mut self) -> &mut TimingTracker {
        &mut self.timing_tracker
    }

    /// Gets read-only access to the timing tracker for statistics.
    ///
    /// # Returns
    ///
    /// An immutable reference to the internal `TimingTracker`
    pub fn timing_tracker(&self) -> &TimingTracker {
        &self.timing_tracker
    }

    /// Exports both step and lambda timings to files with a common prefix.
    ///
    /// This convenience method exports both types of timing data using the
    /// provided prefix, appending "_steps.csv" and "_lambdas.csv" respectively.
    ///
    /// # Arguments
    ///
    /// * `filename_prefix` - Common prefix for the output files
    ///
    /// # Returns
    ///
    /// * `Ok(())` if both exports succeed
    /// * `Err(LassoError)` if either export fails
    ///
    /// # Files Created
    ///
    /// * `{filename_prefix}_steps.csv` - Step timing data
    /// * `{filename_prefix}_lambdas.csv` - Lambda timing data
    pub fn export_all_timings(&self, filename_prefix: &str) -> Result<(), LassoError> {
        let step_filename = format!("{}_steps.csv", filename_prefix);
        let lambda_filename = format!("{}_lambdas.csv", filename_prefix);

        self.export_step_timings(&step_filename)?;
        self.export_lambda_timings(&lambda_filename)?;

        println!("Exported step timings to: {}", step_filename);
        println!("Exported lambda timings to: {}", lambda_filename);

        Ok(())
    }

    /// Prints timing statistics to the console in a formatted summary.
    ///
    /// This method outputs comprehensive timing information including:
    /// - Average, maximum, and count for each ADMM step
    /// - Average, maximum, and count for each Lambda function call
    ///
    /// The output is formatted for easy reading and performance analysis.
    pub fn print_timing_summary(&self) {
        println!("\n=== ADMM Step Timing Summary ===");
        let step_stats = self.timing_tracker.get_step_statistics();
        for (step, (avg, max, count)) in step_stats {
            println!(
                "{}: avg={:.2}ms, max={:.2}ms, count={}",
                step, avg, max, count
            );
        }

        println!("\n=== Lambda Timing Summary ===");
        let lambda_stats = self.timing_tracker.get_lambda_statistics();
        for (lambda, (avg, max, count)) in lambda_stats {
            println!(
                "{}: avg={:.2}ms, max={:.2}ms, count={}",
                lambda, avg, max, count
            );
        }
        println!();
    }
}
