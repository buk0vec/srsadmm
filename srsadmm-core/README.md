# srsadmm-core

![Crates.io Version](https://img.shields.io/crates/v/srsadmm-core)
![docs.rs](https://img.shields.io/docsrs/srsadmm-core)

This is the core library containing the distributed serverless ADMM algorithm, as well as binaries to solve a LASSO regression problem. It is meant for usage with the `tokio` runtime and the `srsadmm-lambda-mm` AWS Lambda function.

## Documentation

[docs.rs](https://docs.rs/srsadmm-core)

## Binaries

- Run the `generate_problem` binary to generate a problem instance.
- Run the `lasso` binary to solve the problem instance, or `lasso_prox` to solve the problem instance with the proximal gradient method. The binaries will pull your AWS credentials from the default environment variables. Please deploy the `srsadmm-lambda-mm` Lambda function to your AWS account first and ensure that it has access to the S3 bucket you are using. To enable a specific backend, run the binary as `cargo run --release --bin <binary> --no-default-features --features <accelerate/netlib/openblas> -- <args>`. 

## Features

- `accelerate` - Use the `accelerate` backend for matrix operations
- `netlib` - Use the `netlib` backend for matrix operations
- `openblas` - Use the `openblas` backend for matrix operations

## Usage

Alternatively, you can use the `srsadmm-core` library in your own project. While you can technically install it with `cargo add srsadmm-core`, it might be better to directly copy this directory into your own project and use it as a dependency. Take a look at `lasso.rs` and `lasso_prox.rs` for examples of how to use the library. You can also install with features `accelerate`, `netlib`, or `openblas` to enable different backends.

### Core ADMM Framework

**`ADMMProblem<G, S>`** - The main trait defining the ADMM algorithm interface. Implementations must provide methods for:
- `precompute()` - One-time setup and matrix factorizations
- `update_x()` - Primal variable update step
- `update_z()` - Auxiliary variable update (often with proximal operators)
- `update_y()` - Dual variable update step
- `update_residuals()` - Compute convergence metrics
- `check_stopping_criteria()` - Determine if algorithm should terminate

**`ADMMSolver<G, S, P>`** - Orchestrates the iterative ADMM optimization process with timing tracking, iteration control, and result export capabilities.

**`ADMMContext<G, S>`** - Execution context containing shared global state (`G`) and local subproblem state (`S`) with thread-safe synchronization primitives.

### Distributed Matrix Variables

**`MatrixVariable`** - A distributed matrix that can be stored and synchronized across multiple backends (local disk, S3, memory). Provides high-level matrix operations for ADMM algorithms while handling distributed storage complexity. Key features:
- Multi-location storage (Local, S3, Memory) with automatic synchronization
- Lazy loading and caching strategies
- Matrix operations: addition, subtraction, multiplication, inversion
- Subproblem-aware row-wise operations with parallel processing

**`DataMatrixVariable`** - Specialized for large read-only matrices (like training data) with memory-mapped file access and efficient chunking for subproblem processing.

**`ScalarVariable`** - Similar distributed storage for scalar values with the same multi-backend synchronization.

### Storage and Resource Management

**`ResourceLocation`** - Enum defining storage backends:
- `Local` - Compressed local filesystem storage
- `S3` - AWS S3 cloud storage for distributed access
- `Memory` - In-memory storage for fast access

**`StorageConfig`** - Configuration for all storage backends with settings for local paths, S3 buckets, and memory management.

**`ProblemResourceImpl<T>`** - Internal resource manager handling storage, synchronization, and caching across multiple backends with automatic consistency management.

### Matrix Operations

**`ops`** module provides distributed matrix operations:
- `mm()` - Matrix multiplication (local or cloud-based via AWS Lambda)
- `lasso_factor()` - Computes (A^T A + ρI)^-1 for LASSO problems
- `soft_threshold()` - L1 regularization proximal operator
- `scale()` - In-place matrix scaling

### Performance and Monitoring

**`TimingTracker`** - Performance tracking for ADMM iterations, outputting a CSV file with the timing data.

### Subproblem Management

**`subproblem`** module provides utilities for decomposing problems:
- `split_matrix_into_subproblems()` - Partitions matrices into row-wise chunks for parallel processing
- `combine_subproblems()` - Reassembles subproblem results into final solution
