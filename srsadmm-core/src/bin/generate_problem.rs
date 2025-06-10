/*
This program generates a LASSO problem and stores the data in the local filesystem.
*/

extern crate nalgebra as na;

use std::path::Path;

use srsadmm_core::{
    resource::{LocalConfig, S3Config, StorageConfig},
    utils::LassoError,
    variable::{DataMatrixVariable, MatrixStorageType},
};
use rand::prelude::*;
use rand_distr::{Normal, StandardNormal};
#[cfg(feature = "rayon")]
use rayon::prelude::*;
use std::time::Instant;

use clap::Parser;

/// Program to generate data matrices A and b for a LASSO problem.
///
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// The number of rows in A
    #[arg(short, long)]
    m: usize,

    /// The number of columns in A
    #[arg(short, long)]
    n: usize,

    /// The number of non-zero elements in x_true
    #[arg(short, long)]
    k: usize,

    /// The number of threads to use for parallelization
    #[arg(short, long, default_value_t = 1)]
    threads: usize,

    /// The prefix for the output files
    #[arg(short, long, default_value_t = ("lasso-large-16gb").to_string())]
    prefix: String,

}

#[tokio::main]
async fn main() {
    let args = Args::parse();

    let m = args.m;
    let n = args.n;
    let num_initial = args.k;
    #[cfg(feature = "rayon")]
    let threads = args.threads;

    let storage_config = StorageConfig::new(
        LocalConfig::new(Path::new("data"), ""),
        S3Config::new("srsadmm", "us-east-1", ""),
    );

    let data_gen_start_time = Instant::now();

    // Pre-allocate the matrix
    let mut a_full_nalgebra = na::DMatrix::<f32>::zeros(m, n);

    let mut rng_seeds = Vec::with_capacity(n);
    for i in 0..n {
        rng_seeds.push(42 + i as u64);
    }

    let a_gen_start = Instant::now();

    let mut rng = rand::rngs::SmallRng::seed_from_u64(42);

    println!("[Main] Generating matrix A...");

    #[cfg(feature = "rayon")]
    {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("Failed to create thread pool");

        pool.install(|| {
            // Generate columns in parallel with optimized batch operations
            let columns: Vec<Vec<f32>> = (0..n)
                .into_par_iter()
                .map(|j| {
                    let mut thread_rng = rand::rngs::SmallRng::seed_from_u64(rng_seeds[j]);

                    // Generate batch of random values for better performance
                    let mut random_values: Vec<f32> =
                        (0..m).map(|_| thread_rng.sample(StandardNormal)).collect();

                    // Compute norm and normalize in-place for better cache usage
                    let norm = random_values.iter().map(|x| x * x).sum::<f32>().sqrt();
                    if norm > 0.0 {
                        random_values.iter_mut().for_each(|x| *x /= norm);
                    }

                    random_values
                })
                .collect();

            // Copy columns into matrix using vectorized operations
            for (j, column_data) in columns.into_iter().enumerate() {
                a_full_nalgebra.column_mut(j).copy_from_slice(&column_data);
            }
        });
    }

    #[cfg(not(feature = "rayon"))]
    {
        // Sequential fallback when rayon is not available
        for j in 0..n {
            let mut thread_rng = rand::rngs::SmallRng::seed_from_u64(rng_seeds[j]);

            // Generate batch of random values for better performance
            let mut random_values: Vec<f32> =
                (0..m).map(|_| thread_rng.sample(StandardNormal)).collect();

            // Compute norm and normalize in-place for better cache usage
            let norm = random_values.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                random_values.iter_mut().for_each(|x| *x /= norm);
            }

            a_full_nalgebra.column_mut(j).copy_from_slice(&random_values);
        }
    }

    println!(
        "[Main] Matrix A generated in {:?}",
        a_gen_start.elapsed()
    );

    // Optimized x_true generation
    let x_true_start = Instant::now();
    let mut x_true_nalgebra = na::DVector::<f32>::zeros(n);
    let indices: Vec<usize> = rand::seq::index::sample(&mut rng, n, num_initial).into_vec();

    // Vectorized assignment for selected indices
    let random_values: Vec<f32> = (0..num_initial)
        .map(|_| rng.sample(StandardNormal))
        .collect();

    for (&idx, &val) in indices.iter().zip(random_values.iter()) {
        x_true_nalgebra[idx] = val;
    }
    println!("[Main] x_true generated in {:?}", x_true_start.elapsed());

    // Optimized noise generation
    let noise_start = Instant::now();
    let noise_dist = Normal::new(0.0, 0.03162)
        .map_err(|e| LassoError::from_string(format!("Failed to create normal dist: {}", e)));
    let noise_dist = noise_dist.expect("Failed to create normal distribution");
    // Generate all noise values at once
    let noise_values: Vec<f32> = (0..m).map(|_| rng.sample(noise_dist)).collect();
    let v_nalgebra = na::DVector::<f32>::from_vec(noise_values);

    println!(
        "[Main] Noise vector generated in {:?}",
        noise_start.elapsed()
    );

    // Matrix-vector operations (these are already optimized in nalgebra)
    let b_computation_start = Instant::now();
    let b_full_nalgebra_vec = &a_full_nalgebra * &x_true_nalgebra + v_nalgebra;
    let b_full_nalgebra = na::DMatrix::from_column_slice(m, 1, b_full_nalgebra_vec.as_slice());
    println!(
        "[Main] b = Ax + v computed in {:?}",
        b_computation_start.elapsed()
    );

    // Calculate ||A^T b||_inf
    let lambda_max = {
      let mut max_abs_val: f32 = 0.0;
      for col in 0..n {
        let col_view = a_full_nalgebra.column(col);
        let val = col_view.dot(&b_full_nalgebra.column(0));
        max_abs_val = max_abs_val.max(val.abs());
      }
      max_abs_val
    };

    println!("[Main] Lambda_max: {}", lambda_max);

    let _ = DataMatrixVariable::from_matrix(
        format!("{}-a", args.prefix),
        a_full_nalgebra,
        MatrixStorageType::Rows,
        &storage_config,
    );
    let _ = DataMatrixVariable::from_matrix(
        format!("{}-b", args.prefix),
        b_full_nalgebra,
        MatrixStorageType::Rows,
        &storage_config,
    );

    println!(
        "[Main] Data generation complete in {:?}",
        data_gen_start_time.elapsed()
    );
}

