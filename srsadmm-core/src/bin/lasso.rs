/*
    lasso.rs
    By Nick Bukovec
    Solves a LASSO problem using serverless matrix multiplication.
*/

use clap::Parser;
use futures::future;
use nalgebra as na;
use rand::prelude::*;
use rand_distr::{Normal, StandardNormal};
#[cfg(feature = "rayon")]
use rayon::prelude::*;
use srsadmm_core::utils::{LassoError, fused_lasso_factor};
use srsadmm_core::variable::{DataMatrixVariable, MatrixStorageType};
use srsadmm_core::{
    ops::mm,
    problem::{ADMMConfig, ADMMContext},
    problem::{ADMMProblem, ADMMSolver},
    resource::{LocalConfig, MemoryConfig, ResourceLocation, S3Config, StorageConfig},
    variable::MatrixVariable,
};
use std::error::Error;
use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::Semaphore;
use tokio::sync::{Mutex, RwLock};
use tokio::task::JoinHandle;

/// Program to solve a LASSO problem using ADMM.
/// You can either provide a a_file and b_file in the format generated by generate_problem.rs,
/// or generate a problem instance on-the-fly with n, m, and k.
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
pub struct Args {
    /// The number of threads to use for parallelization
    #[arg(short, long, default_value_t = 8)]
    threads: usize,

    /// The prefix for the output files
    #[arg(short, long, default_value_t = ("lasso").to_string())]
    prefix: String,

    /// The number of subproblems to use
    #[arg(short, long, default_value_t = 40)]
    subproblems: usize,

    /// The number of iterations to run
    #[arg(short, long, default_value_t = 50)]
    iterations: usize,

    /// File path to preprocessed data matrix A (see generate_problem.rs)
    #[arg(short, long)]
    a_file: Option<String>,

    /// File path to preprocessed data matrix b (see generate_problem.rs)
    #[arg(short, long)]
    b_file: Option<String>,

    /// The number of features to use if generating a problem instance
    #[arg(short, long, default_value_t = 10_000)]
    n: usize,

    /// The number of samples to use if generating a problem instance
    #[arg(short, long, default_value_t = 400_000)]
    m: usize,

    /// The number of non-zero elements to use if generating a problem instance
    #[arg(short, long, default_value_t = 1000)]
    k: usize,

    /// ADMM rho parameter
    #[arg(short, long, default_value_t = 0.25)]
    rho: f32,

    /// ADMM absolute tolerance
    #[arg(short, long, default_value_t = 1e-4)]
    abs_eps: f32,

    /// ADMM relative tolerance
    #[arg(short, long, default_value_t = 1e-2)]
    rel_eps: f32,

    /// ADMM lambda parameter
    #[arg(short, long, default_value_t = 0.32)]
    lambda: f32,

    /// S3 bucket to use for storage
    #[arg(short, long, default_value_t = ("srsadmm").to_string())]
    s3_bucket: String,
}

// Context for LASSO problem
pub struct LassoContext {
    pub a: DataMatrixVariable,
    pub b: DataMatrixVariable,
    pub z: MatrixVariable,
    pub z_old: MatrixVariable,
    pub primal_residuals: Vec<f32>,
    pub dual_residuals: Vec<f32>,
    pub x_norm: f32,
    pub y_norm: f32,
    pub z_norm: f32,
    pub args: Args, // Add args to the context
}

// Subproblem state
#[derive(Clone)]
pub struct LassoSubproblem {
    /// A_i^T * A_i + ρ * I
    pub at_a_rho_i_inv: MatrixVariable,
    /// A_i^T * b
    pub at_b: MatrixVariable,
    /// x_i
    pub x: MatrixVariable,
    /// y_i
    pub y: MatrixVariable,
}

impl LassoContext {
    pub fn new(
        a: DataMatrixVariable,
        b: DataMatrixVariable,
        storage_config: &StorageConfig,
        args: Args,
    ) -> Self {
        LassoContext {
            a,
            b,
            // Store z and z_old in memory for faster access
            z: MatrixVariable::new("z".to_string(), ResourceLocation::Memory, storage_config),
            z_old: MatrixVariable::new(
                "z_old".to_string(),
                ResourceLocation::Memory,
                storage_config,
            ),
            primal_residuals: Vec::new(),
            dual_residuals: Vec::new(),
            x_norm: 0.0,
            y_norm: 0.0,
            z_norm: 0.0,
            args,
        }
    }

    fn n_subproblems(&self) -> usize {
        self.args.subproblems
    }

    fn max_iterations(&self) -> usize {
        self.args.iterations
    }

    fn absolute_tolerance(&self) -> f32 {
        self.args.abs_eps
    }

    fn relative_tolerance(&self) -> f32 {
        self.args.rel_eps
    }

    fn rho(&self) -> f32 {
        self.args.rho
    }

    fn lambda(&self) -> f32 {
        self.args.lambda
    }

    fn n_features(&self) -> usize {
        self.args.n
    }

    fn n_threads(&self) -> usize {
        self.args.threads
    }
}

// LASSO problem implementation
pub struct LassoProblem {
    context: Arc<ADMMContext<LassoContext, Vec<LassoSubproblem>>>,
}

impl LassoProblem {
    pub fn new(context: Arc<ADMMContext<LassoContext, Vec<LassoSubproblem>>>) -> Self {
        LassoProblem { context }
    }
}

impl ADMMProblem<LassoContext, Vec<LassoSubproblem>> for LassoProblem {
    fn context(&self) -> &ADMMContext<LassoContext, Vec<LassoSubproblem>> {
        &self.context
    }
    // Precompute the subproblems
    async fn precompute(&self) -> Result<(), LassoError> {
        let precompute_start_time = Instant::now();
        println!("[LassoProblem] Starting precomputation...");
        let mut context_global = self.context.global.write().await;

        let n_features = context_global.n_features();
        let n_subproblems = context_global.n_subproblems();
        let n_threads = context_global.n_threads();
        let rho = context_global.rho();
        // Only allow n_threads to run at a time
        let semaphore = Arc::new(Semaphore::new(n_threads));

        // Initialize z_old and z with zeros
        let z_old_initial_matrix = na::DMatrix::<f32>::zeros(n_features, 1);
        context_global
            .z_old
            .write(z_old_initial_matrix)
            .await
            .map_err(|e| LassoError::from_string(format!("Failed to initialize z_old: {}", e)))?;
        let z_initial_matrix = na::DMatrix::<f32>::zeros(n_features, 1);
        context_global
            .z
            .write(z_initial_matrix)
            .await
            .map_err(|e| LassoError::from_string(format!("Failed to initialize z: {}", e)))?;

        // Split A and b into subproblems
        println!("[LassoProblem] Starting to map rows by subproblem");

        // Iterate over chunks of both A and b to create subproblems
        let futures = context_global
            .a
            .iter_subproblems_with(&context_global.b, n_subproblems, |a_chunk, b_chunk, i| {
                let storage_config = self.context.config.storage.clone();
                // Control the number of threads running at a time
                let semaphore = semaphore.clone();
                tokio::spawn(async move {
                    // Acquire semaphore
                    let permit = semaphore.clone().acquire_owned().await.unwrap();
                    println!("[LassoProblem] [{}] Factoring subproblem", i);
                    // Move CPU-intensive computation to blocking thread pool
                    let (a_t_a_rho_i_inv, a_t_b) = tokio::task::spawn_blocking(move || {
                        fused_lasso_factor(&a_chunk, &b_chunk, rho)
                    })
                    .await
                    .expect("Blocking task failed");
                    println!(
                        "[LassoProblem] [{}] Factoring subproblem done, writing to S3",
                        i
                    );
                    // Write A_i^T * A_i + ρ * I to S3
                    let at_a_rho_i_inv = MatrixVariable::from_matrix(
                        format!("subproblem-{}-at_a_rho_i_inv", i),
                        ResourceLocation::S3,
                        &storage_config,
                        a_t_a_rho_i_inv,
                    )
                    .await;
                    // Variable for x_i stored in S3
                    let x = MatrixVariable::zeros(
                        format!("subproblem-{}-x", i),
                        n_features,
                        1,
                        ResourceLocation::S3,
                        &storage_config,
                    )
                    .await;
                    println!(
                        "[LassoProblem] [{}] Writing to S3 done, writing to local",
                        i
                    );
                    // Write A_i^T * b to local FS
                    let at_b = MatrixVariable::from_matrix(
                        format!("subproblem-{}-atb", i),
                        ResourceLocation::Local,
                        &storage_config,
                        a_t_b,
                    )
                    .await;
                    // Variable for y_i stored in local FS
                    let y = MatrixVariable::zeros(
                        format!("subproblem-{}-y", i),
                        n_features,
                        1,
                        ResourceLocation::Local,
                        &storage_config,
                    )
                    .await;
                    drop(permit); // Release semaphore
                    println!("[LassoProblem] [{}] Completed", i);
                    LassoSubproblem {
                        at_a_rho_i_inv,
                        at_b,
                        x,
                        y,
                    }
                })
            })
            .expect("Error when generating futures");

        // Wait for all subproblems to be written to S3 and local FS
        let results = future::try_join_all(futures).await;

        println!(
            "[LassoProblem] Finished writing subproblems to S3 and local in {:?}",
            precompute_start_time.elapsed()
        );
        // Store results in local context
        let mut local = self.context.local.lock().await;
        local.clear();
        local.extend(results.unwrap());

        // Print the time it took to precompute the subproblems
        println!(
            "[LassoProblem] Precomputation completed in {:?}",
            precompute_start_time.elapsed()
        );
        Ok(())
    }

    // x-update step
    async fn update_x(&self) -> Result<(), LassoError> {
        // Read the global context
        let context_global = self.context.global.read().await;
        // Read the local context
        let local_guard = self.context.local.lock().await;

        // Get the number of subproblems and rho
        let n_subproblems = context_global.n_subproblems();
        let rho = context_global.rho();

        // Create futures for parallel x-update
        let mut futures = Vec::with_capacity(n_subproblems);

        // Iterate over the subproblems
        for i in 0..n_subproblems {
            // Get the storage config and variables
            let storage_config = self.context.config.storage.clone();
            let z_var = context_global.z.clone();
            let y_var = local_guard[i].y.clone();
            let at_b_var = local_guard[i].at_b.clone();
            let at_a_rho_i_inv_var = local_guard[i].at_a_rho_i_inv.clone();
            let x_var = local_guard[i].x.clone();

            // Spawn a new task for the x-update
            let handle = tokio::spawn(async move {
                // Read all required matrices from local FS
                let z_matrix = z_var.read().await.map_err(|e| {
                    LassoError::from_string(format!("[Subproblem {}] Failed to read z: {}", i, e))
                })?;
                let y_matrix = y_var.read().await.map_err(|e| {
                    LassoError::from_string(format!("[Subproblem {}] Failed to read y: {}", i, e))
                })?;
                let at_b_matrix = at_b_var.read().await.map_err(|e| {
                    LassoError::from_string(format!(
                        "[Subproblem {}] Failed to read at_b: {}",
                        i, e
                    ))
                })?;

                // Compute rho * (z - y) locally
                let rho_z_minus_y = rho * (z_matrix - y_matrix);
                let x_update_vector = at_b_matrix + rho_z_minus_y;

                // Create temporary variable for the combined A^T*b + ρ*(z - y) on S3
                let mut atb_rho_z_y = MatrixVariable::from_matrix(
                    format!("subproblem-{}-atb-rho-z-y", i),
                    ResourceLocation::S3, // Changed to S3 to trigger lambda
                    &storage_config,
                    x_update_vector,
                )
                .await;
                // Create temporary mutable variable for the result of lambda call
                let mut x_result = x_var.clone();
                // Clone the precomputed inverse for lambda call
                let mut at_a_rho_i_inv_clone = at_a_rho_i_inv_var.clone();

                // This will trigger a lambda call and write result to S3
                mm(&mut at_a_rho_i_inv_clone, &mut atb_rho_z_y, &mut x_result)
                    .await
                    .map_err(|e| {
                        LassoError::from_string(format!(
                            "[Subproblem {}] mm x_update failed: {}",
                            i, e
                        ))
                    })?;

                Ok::<(), LassoError>(())
            });
            // Add the handle to the vector of futures
            futures.push(handle);
        }

        // Wait for all x-updates to complete
        let results = future::join_all(futures).await;

        // Check for errors
        for (i, result) in results.into_iter().enumerate() {
            match result {
                Ok(Ok(())) => {}
                Ok(Err(e)) => return Err(e),
                Err(join_error) => {
                    return Err(LassoError::from_string(format!(
                        "Error in x-update task {}: {}",
                        i, join_error
                    )));
                }
            }
        }

        Ok(())
    }

    // z-update step
    async fn update_z(&self) -> Result<(), LassoError> {
        // Read the global context
        let mut context_global = self.context.global.write().await;
        // Read the local context
        let local_guard = self.context.local.lock().await;

        let n_features = context_global.n_features();
        let n_subproblems = context_global.n_subproblems();
        let rho = context_global.rho();
        let lambda = context_global.lambda();

        // Before updating z, store current z as z_old
        let current_z_matrix_for_old = context_global.z.read().await.map_err(|e| {
            LassoError::from_string(format!("Failed to read z for z_old backup: {}", e))
        })?;
        context_global
            .z_old
            .write(current_z_matrix_for_old)
            .await
            .map_err(|e| LassoError::from_string(format!("Failed to write z_old: {}", e)))?;

        // Initialize the sum of x and y
        let mut sum_x = na::DMatrix::<f32>::zeros(n_features, 1);
        let mut sum_y = na::DMatrix::<f32>::zeros(n_features, 1);

        // Read all subproblem variables
        let futures: Vec<JoinHandle<Result<(na::DMatrix<f32>, na::DMatrix<f32>), LassoError>>> =
            local_guard
                .iter()
                .enumerate()
                .map(|(idx, subproblem)| {
                    // Clone the variables so they can be moved into the spawn
                    let mut x_var = subproblem.x.clone();
                    let y_var = subproblem.y.clone();

                    tokio::spawn(async move {
                        // Cache x in local FS when reading it for the future y-update
                        let x_matrix = x_var
                            .read_and_cache(ResourceLocation::Local)
                            .await
                            .map_err(|e| {
                                LassoError::from_string(format!(
                                    "[Subproblem {}] read subproblem.x for z-update failed: {}",
                                    idx, e
                                ))
                            })?;
                        // Read y matrix, it's already cached in local FS
                        let y_matrix = y_var.read().await.map_err(|e| {
                            LassoError::from_string(format!(
                                "[Subproblem {}] read subproblem.y for z-update failed: {}",
                                idx, e
                            ))
                        })?;
                        Ok((x_matrix, y_matrix))
                    })
                })
                .collect();

        // Wait for all reads to complete and sum the results
        let results = futures::future::try_join_all(futures).await.unwrap();
        for result in results {
            let (x_matrix, y_matrix) = result.unwrap();
            sum_x += x_matrix;
            sum_y += y_matrix;
        }

        // Compute the average of x and y
        let x_bar = sum_x / (n_subproblems as f32);
        let y_bar = sum_y / (n_subproblems as f32);
        let x_plus_y_bar = x_bar + y_bar;

        // Compute the soft threshold
        let threshold_val = lambda / (rho * n_subproblems as f32);
        let x_plus_y_bar_sign = x_plus_y_bar.map(|x| {
            if x > 0.0 {
                1.0
            } else if x < 0.0 {
                -1.0
            } else {
                0.0
            }
        });

        let z_new_matrix = x_plus_y_bar_sign.map_with_location(|i, j, x_s| {
            x_s * (x_plus_y_bar[(i, j)].abs() - threshold_val).max(0.0)
        });

        // Write the new z back to FS
        context_global.z.write(z_new_matrix).await.map_err(|e| {
            LassoError::from_string(format!("write context_global.z failed: {}", e))
        })?;
        Ok(())
    }

    // y-update step
    async fn update_y(&self) -> Result<(), LassoError> {
        // Read the global context
        let context_global = self.context.global.read().await;
        // Read the local context
        let local_guard = self.context.local.lock().await;

        let n_subproblems = context_global.n_subproblems();

        // Create futures for parallel y-update
        let mut futures = Vec::with_capacity(n_subproblems);

        // Iterate over the subproblems
        for i in 0..n_subproblems {
            let z_var = context_global.z.clone();
            let x_var = local_guard[i].x.clone();
            let y_var = local_guard[i].y.clone();

            // New thread for the y-update for each subproblem
            let handle = tokio::spawn(async move {
                // Read required matrices from local FS
                let z_matrix = z_var.read().await.map_err(|e| {
                    LassoError::from_string(format!(
                        "[Subproblem {}] read z for y-update failed: {}",
                        i, e
                    ))
                })?;
                let x_matrix = x_var.read().await.map_err(|e| {
                    LassoError::from_string(format!(
                        "[Subproblem {}] read subproblem.x for y-update failed: {}",
                        i, e
                    ))
                })?;
                let y_matrix = y_var.read().await.map_err(|e| {
                    LassoError::from_string(format!(
                        "[Subproblem {}] read subproblem.y for y-update failed: {}",
                        i, e
                    ))
                })?;

                // Compute the y-update
                let y_update_matrix = y_matrix + (x_matrix - z_matrix); // y + (x - z)

                // Write the new y back to FS
                let mut y_var_mut = y_var.clone();
                y_var_mut.write(y_update_matrix).await.map_err(|e| {
                    LassoError::from_string(format!(
                        "[Subproblem {}] write subproblem.y failed: {}",
                        i, e
                    ))
                })?;

                Ok::<(), LassoError>(())
            });
            futures.push(handle);
        }

        // Wait for all y-updates to complete
        let results = future::join_all(futures).await;

        // Check for errors
        for (i, result) in results.into_iter().enumerate() {
            match result {
                Ok(Ok(())) => {}
                Ok(Err(e)) => return Err(e),
                Err(join_error) => {
                    return Err(LassoError::from_string(format!(
                        "Error in y-update task {}: {}",
                        i, join_error
                    )));
                }
            }
        }

        Ok(())
    }

    // Update the residuals
    async fn update_residuals(&self) -> Result<(), LassoError> {
        // Read the global context
        let mut context_global = self.context.global.write().await;
        // Read the local context
        let mut local_guard = self.context.local.lock().await;

        let mut squared_resids_sum: f32 = 0.0;
        let mut x_norm_squared_sum: f32 = 0.0;
        let mut y_norm_squared_sum: f32 = 0.0;

        // Read the current z
        let z_k_plus_1_matrix =
            context_global.z.read().await.map_err(|e| {
                LassoError::from_string(format!("read context_global.z failed: {}", e))
            })?;
        let current_z_norm = z_k_plus_1_matrix.norm();

        // Read all subproblem variables (no sync needed, they're in their default locations)
        for (idx, subproblem) in local_guard.iter_mut().enumerate() {
            let x_matrix = subproblem.x.read().await.map_err(|e| {
                LassoError::from_string(format!(
                    "[Subproblem {}] read subproblem.x for residual failed: {}",
                    idx, e
                ))
            })?;
            let y_matrix = subproblem.y.read().await.map_err(|e| {
                LassoError::from_string(format!(
                    "[Subproblem {}] read subproblem.y for residual failed: {}",
                    idx, e
                ))
            })?;

            // Compute the residuals
            squared_resids_sum += (&x_matrix - &z_k_plus_1_matrix).norm_squared();
            x_norm_squared_sum += x_matrix.norm_squared();
            y_norm_squared_sum += y_matrix.norm_squared();
        }

        // Compute the norms
        let total_x_norm: f32 = x_norm_squared_sum.sqrt();
        let total_y_norm: f32 = y_norm_squared_sum.sqrt();

        // Write the norms back to the global context
        context_global.x_norm = total_x_norm;
        context_global.y_norm = total_y_norm;
        context_global.z_norm = current_z_norm;

        // Compute the primal residual
        let primal_residual: f32 = squared_resids_sum.sqrt();
        context_global.primal_residuals.push(primal_residual);

        // Read the old z
        let z_k_matrix = context_global.z_old.read().await.map_err(|e| {
            LassoError::from_string(format!("Failed to read z_old for dual_residual: {}", e))
        })?;
        // Compute the difference between the new and old z
        let z_diff = z_k_plus_1_matrix.clone() - z_k_matrix.clone();
        // Compute the norm of the difference
        let z_diff_norm = z_diff.norm();
        // Compute the dual residual
        // RHO * (z_k+1 - z_k).norm()
        let rho = context_global.rho();
        let actual_dual_residual = rho * z_diff_norm;
        context_global.dual_residuals.push(actual_dual_residual);

        println!(
            "[LassoProblem] Primal residual: {:.4e}, Dual residual: {:.4e}",
            primal_residual, actual_dual_residual
        );
        Ok(())
    }

    // Check the stopping criteria
    async fn check_stopping_criteria(&self) -> Result<bool, LassoError> {
        let context = self.context.global.read().await;

        // Get the last primal residual
        let primal_residual = context.primal_residuals.last().ok_or_else(|| {
            LassoError::from_string("Primal residuals vector is empty".to_string())
        })?;
        // Get the last dual residual
        let dual_residual = context
            .dual_residuals
            .last()
            .ok_or_else(|| LassoError::from_string("Dual residuals vector is empty".to_string()))?;

        let n_features = context.n_features();
        let n_subproblems = context.n_subproblems();
        let abs_eps = context.absolute_tolerance();
        let rel_eps = context.relative_tolerance();
        let rho = context.rho();

        // Compute the primal stopping criterion
        let eps_primal = abs_eps * (n_features as f32).sqrt() * (n_subproblems as f32).sqrt()
            + rel_eps
                * context
                    .x_norm
                    .max(context.z_norm * (n_subproblems as f32).sqrt());
        // Compute the dual stopping criterion
        let eps_dual =
            abs_eps * ((n_subproblems * n_features) as f32).sqrt() + rel_eps * rho * context.y_norm;

        // Check if the stopping criteria are met
        let should_stop = *primal_residual <= eps_primal && *dual_residual <= eps_dual;
        if should_stop {
            println!(
                "[LassoProblem] CONVERGED! Primal_res: {:.4e} <= {:.4e}, Dual_res: {:.4e} <= {:.4e}",
                primal_residual, eps_primal, dual_residual, eps_dual
            );
        }
        // Return whether the stopping criteria are met
        Ok(should_stop)
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    println!("[Main] LASSO starting up...");
    let mut rng = rand::rngs::SmallRng::seed_from_u64(0);

    let mut args = Args::parse();

    let prefix = args.prefix.clone();
    let a_file = args.a_file.clone();
    let b_file = args.b_file.clone();
    let n = args.n;
    let m = args.m;
    let k = args.k;

    // Set up storage configuration
    let storage_config = StorageConfig {
        local: LocalConfig {
            root: "./tmp".to_string(),
            prefix: prefix.clone(),
        },
        s3: S3Config {
            bucket: args.s3_bucket.clone(),
            // This actually doesn't matter, the AWS client uses env vars to determine the region
            region: "us-west-2".to_string(),
            prefix: prefix.clone(),
        },
        memory: MemoryConfig::new(),
    };

    let (a, b) = if a_file.is_some() && b_file.is_some() {
        let a = DataMatrixVariable::from_problem_file(
            'a'.to_string(),
            Path::new(&a_file.unwrap()),
            &storage_config,
        );
        let b = DataMatrixVariable::from_problem_file(
            'b'.to_string(),
            Path::new(&b_file.unwrap()),
            &storage_config,
        );
        (a, b)
    } else {
        println!("[Main] Starting data generation...");
        let data_gen_start_time = Instant::now();

        // Optimized matrix generation using parallel column generation
        println!("[Main] Generating matrix A using parallel processing...");
        let a_gen_start = Instant::now();

        // Pre-allocate the matrix
        let mut a_full_nalgebra = na::DMatrix::<f32>::zeros(m, n);

        let mut rng_seeds = Vec::with_capacity(n);
        for i in 0..n {
            rng_seeds.push(42 + i as u64);
        }

        // Use parallel iterator to generate columns
        #[cfg(feature = "rayon")]
        let columns: Vec<na::DVector<f32>> = (0..n)
            .into_par_iter()
            .map(|i| {
                // Each thread gets its own RNG to avoid contention
                let mut thread_rng = rand::rngs::SmallRng::seed_from_u64(rng_seeds[i]);

                // Generate the entire column at once
                let mut column = na::DVector::<f32>::zeros(m);
                for i in 0..m {
                    column[i] = thread_rng.sample(StandardNormal);
                }

                // Normalize the column to unit length
                let norm = column.norm();
                if norm > 0.0 {
                    column /= norm;
                }
                column
            })
            .collect();

        // Sequential fallback when rayon is not available
        #[cfg(not(feature = "rayon"))]
        let columns: Vec<na::DVector<f32>> = (0..n)
            .map(|i| {
                // Each thread gets its own RNG to avoid contention
                let mut thread_rng = rand::rngs::SmallRng::seed_from_u64(rng_seeds[i]);

                // Generate the entire column at once
                let mut column = na::DVector::<f32>::zeros(m);
                for i in 0..m {
                    column[i] = thread_rng.sample(StandardNormal);
                }

                // Normalize the column to unit length
                let norm = column.norm();
                if norm > 0.0 {
                    column /= norm;
                }
                column
            })
            .collect();

        // Copy columns into the matrix (this is fast, sequential memory access)
        for (j, column) in columns.into_iter().enumerate() {
            a_full_nalgebra.set_column(j, &column);
        }

        println!(
            "[Main] Matrix A generated in {:?}",
            a_gen_start.elapsed()
        );

        // Optimized x_true generation
        let x_true_start = Instant::now();
        let mut x_true_nalgebra = na::DVector::<f32>::zeros(n);
        let indices: Vec<usize> = rand::seq::index::sample(&mut rng, n, k).into_vec();

        // Vectorized assignment for selected indices
        let random_values: Vec<f32> = (0..k).map(|_| rng.sample(StandardNormal)).collect();

        for (&idx, &val) in indices.iter().zip(random_values.iter()) {
            x_true_nalgebra[idx] = val;
        }
        println!("[Main] x_true generated in {:?}", x_true_start.elapsed());

        // Optimized noise generation
        let noise_start = Instant::now();
        let noise_dist = Normal::new(0.0, 0.03162)
            .map_err(|e| LassoError::from_string(format!("Failed to create normal dist: {}", e)))?;

        // Generate all noise values at once
        let noise_values: Vec<f32> = (0..m).map(|_| rng.sample(noise_dist)).collect();
        let v_nalgebra = na::DVector::<f32>::from_vec(noise_values);

        println!(
            "[Main] Noise vector generated in {:?}",
            noise_start.elapsed()
        );

        // Compute b = Ax + v
        let b_computation_start = Instant::now();
        let b_full_nalgebra_vec = &a_full_nalgebra * &x_true_nalgebra + v_nalgebra;
        let b_full_nalgebra = na::DMatrix::from_column_slice(m, 1, b_full_nalgebra_vec.as_slice());
        println!(
            "[Main] b = Ax + v computed in {:?}",
            b_computation_start.elapsed()
        );

        // Uncomment to print lambda_max
        // println!(
        //     "[Main] Data generation complete in {:?}. Lambda_max: {}",
        //     data_gen_start_time.elapsed(),
        //     (&a_full_nalgebra.transpose() * &b_full_nalgebra)
        //         .abs()
        //         .max()
        // );

        println!(
            "[Main] Data generation complete in {:?}",
            data_gen_start_time.elapsed()
        );

        println!("[Main] Writing A and b to DataMatrixVariables...");
        let a = DataMatrixVariable::from_matrix(
            'a'.to_string(),
            a_full_nalgebra,
            MatrixStorageType::Rows,
            &storage_config,
        );
        println!("[Main] A written to DataMatrixVariable.");
        let b = DataMatrixVariable::from_matrix(
            'b'.to_string(),
            b_full_nalgebra,
            MatrixStorageType::Rows,
            &storage_config,
        );
        println!("[Main] B written to DataMatrixVariable.");
        (a, b)
    };

    args.n = a.ncols();
    args.m = a.nrows();

    println!("[Main] n: {}, m: {}", args.n, args.m);

    // Create context
    let context = Arc::new(ADMMContext {
        config: ADMMConfig {
            storage: storage_config.clone(),
        },
        global: Arc::new(RwLock::new(LassoContext::new(a, b, &storage_config, args))),
        local: Arc::new(Mutex::new(Vec::new())),
    });
    println!("[Main] ADMMContext created.");

    // Create problem and solver
    let problem = LassoProblem::new(context.clone());
    let max_iterations = context.global.read().await.max_iterations();
    let mut solver = ADMMSolver::new(problem, max_iterations);
    println!("[Main] LassoProblem and ADMMSolver created.");

    // Solve
    println!("[Main] Starting ADMM solver...");
    let solver_start_time = Instant::now();
    solver.solve().await?;
    println!(
        "[Main] ADMM solver finished in {:?}.",
        solver_start_time.elapsed()
    );

    let z = context.global.read().await.z.read().await.unwrap();
    // Output z to csv
    let mut file = File::create(format!(
        "{}/{}-final-z.csv",
        context.config.storage.local.root, context.config.storage.local.prefix
    ))
    .unwrap();
    for i in 0..z.ncols() {
        writeln!(
            file,
            "{}",
            z.column(i)
                .to_owned()
                .iter()
                .map(|x| x.to_string())
                .collect::<Vec<String>>()
                .join(",")
        )
        .unwrap();
    }
    file.flush().unwrap();

    println!(
        "[Main] lasso.rs finished. Final z is stored in {}",
        format!(
            "{}/{}-final-z.csv",
            context.config.storage.local.root, context.config.storage.local.prefix
        )
    );

    println!("[Main] Timing summary:");
    solver.print_timing_summary();

    println!("[Main] Exporting step timings to CSV...");
    let _ = solver.export_step_timings(
        format!(
            "{}/{}-timings.csv",
            context.config.storage.local.root, context.config.storage.local.prefix
        )
        .as_str(),
    );
    Ok(())
}
