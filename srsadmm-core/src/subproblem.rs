use std::error::Error;
extern crate nalgebra as na;
use crate::resource::{ResourceLocation, StorageConfig};
use crate::variable::MatrixVariable;

/// Splits a matrix variable into chunks of rows for subproblems
///
/// # Arguments
/// * `matrix` - The matrix variable to split
/// * `n_subproblems` - Number of subproblems to split into
/// * `storage_config` - Storage configuration for the new variables
/// * `prefix` - Prefix for the subproblem IDs to avoid conflicts
///
/// # Returns
/// * `Result<Vec<DMatrixVariable>>` - Vector of matrix variables, one for each subproblem
pub async fn split_matrix_into_subproblems(
    matrix: &MatrixVariable,
    n_subproblems: usize,
    storage_config: &StorageConfig,
    default_location: ResourceLocation,
    prefix: &str,
) -> Result<Vec<MatrixVariable>, Box<dyn Error>> {
    // Read the full matrix
    // println!("[Splitter] Reading full matrix for variable ID: {}", matrix.id());
    let full_matrix = matrix.read().await?;
    let nrows = full_matrix.nrows();
    // println!("[Splitter] Full matrix for ID '{}' dimensions: {}x{}", matrix.id(), nrows, ncols);

    // Calculate chunk size and handle remainder
    let base_chunk_size = nrows / n_subproblems;
    let remainder = nrows % n_subproblems;

    let mut subproblems = Vec::with_capacity(n_subproblems);
    let mut current_row = 0;

    for i in 0..n_subproblems {
        println!("[Splitter] Processing subproblem {}/{}", i, n_subproblems);
        // Calculate chunk size for this subproblem
        let chunk_size = if i < remainder {
            base_chunk_size + 1
        } else {
            base_chunk_size
        };

        // Extract the chunk
        let chunk = full_matrix.rows(current_row, chunk_size).clone_owned();

        // Create new variable for this subproblem
        let subproblem_id = format!("subproblem-{}-{}", prefix, i);
        let mut subproblem = MatrixVariable::new(subproblem_id, default_location, storage_config);

        println!(
            "[Splitter] Writing chunk of size {}x{} to subproblem ID: {}",
            &chunk.nrows(),
            &chunk.ncols(),
            subproblem.id()
        );

        // Write the chunk to the new variable
        subproblem.write(chunk).await?;

        subproblems.push(subproblem);
        current_row += chunk_size;
    }

    Ok(subproblems)
}

/// Combines subproblem matrices back into a single matrix
///
/// # Arguments
/// * `subproblems` - Vector of matrix variables from subproblems
/// * `storage_config` - Storage configuration for the combined variable
/// * `id` - ID for the combined variable
///
/// # Returns
/// * `Result<DMatrixVariable>` - Combined matrix variable
pub async fn combine_subproblems(
    subproblems: &[MatrixVariable],
    storage_config: &StorageConfig,
    id: String,
) -> Result<MatrixVariable, Box<dyn Error>> {
    // Read all subproblems
    let mut chunks = Vec::with_capacity(subproblems.len());
    let mut total_rows = 0;
    let ncols = subproblems[0].read().await?.ncols();

    for subproblem in subproblems {
        let chunk = subproblem.read().await?;
        total_rows += chunk.nrows();
        chunks.push(chunk);
    }

    // Create combined matrix
    let mut combined = na::DMatrix::<f32>::zeros(total_rows, ncols);
    let mut current_row = 0;

    for chunk in chunks {
        combined
            .rows_mut(current_row, chunk.nrows())
            .copy_from(&chunk);
        current_row += chunk.nrows();
    }

    // Create and write to new variable
    let mut result = MatrixVariable::new(id, subproblems[0].default_location(), storage_config);
    result.write(combined).await?;

    Ok(result)
}
