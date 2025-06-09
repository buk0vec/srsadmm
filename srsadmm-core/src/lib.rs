//! srsadmm-core
//!
//! This library provides the core functionality for the srsadmm project, including
//! matrix operations, problem formulation, and resource management.
//!
//! # Features
//!
//! - Local and remote state management
//! - Matrix operations
//! - LASSO-specific operations
//! - Problem formulation
//! - Timing and logging

/// Lambda-specific operations 
pub(crate) mod lambda;

/// Matrix operations
pub mod ops;

/// Problem formulation
pub mod problem;

/// Resource management
pub mod resource;

/// S3 storage operations
pub(crate) mod s3;

/// Storage management
pub(crate) mod storage;

/// Subproblem management
pub mod subproblem;

/// Timing and logging
pub mod timing;

/// Utility functions
pub mod utils;

/// Variable management
pub mod variable;
