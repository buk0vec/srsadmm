//! srsadmm-core
//!
//! This library provides the core functionality for the srsadmm project, including
//! matrix operations, problem formulation, and resource management. 
//! It is meant for usage with the `tokio` runtime and the `srsadmm-lambda-mm` AWS Lambda function.
//! For more information, please see the GitHub repository: <https://github.com/buk0vec/srsadmm>.
//!
//! # Functionality
//!
//! - Local and remote state management
//! - Matrix operations
//! - LASSO-specific operations
//! - ADMM problem formulation and solving
//! - Timing and logging
//! 
//! # Features 
//!
//! - `accelerate` - Use the `accelerate` backend for matrix operations
//! - `netlib` - Use the `netlib` backend for matrix operations
//! - `openblas` - Use the `openblas` backend for matrix operations
//! 
//! 

/// Lambda-specific operations
pub(crate) mod lambda;

/// Matrix operations on distributed variables
pub mod ops;

/// Problem formulation and solving
pub mod problem;

/// Resource management for distributed variables
pub mod resource;

/// S3 storage operations
pub(crate) mod s3;

/// Storage management
pub(crate) mod storage;

/// Utility functions for splitting and combining distributed variables
pub mod subproblem;

/// Timing and logging utilities
pub mod timing;

/// Utility functions for LASSO problems
pub mod utils;

/// Distributed variable management
pub mod variable;
