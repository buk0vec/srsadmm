/// Operations on variables.
extern crate nalgebra as na;

use crate::lambda::{
    call_factor_lambda_logged,
    call_matrix_multiplication_lambda_logged_with_metadata,
};
use crate::resource::ResourceLocation;
use crate::utils::fast_lasso_inverse;
use crate::variable::MatrixVariable;
use aws_sdk_lambda::Client as LambdaClient;

/// Matrix multiplication function, writing the output to dest.
/// Executes either locally or in the cloud, depending on the destination location.
///
/// # Arguments
///
/// * `a` - The first matrix to multiply.
/// * `b` - The second matrix to multiply.
/// * `dest` - The destination matrix to write the result to.
///
/// # Returns
///
/// * `Ok(())` if the operation is successful.
/// * `Err(Box<dyn std::error::Error>)` if the operation fails.
///
/// # Example
///
pub async fn mm(
    a: &mut MatrixVariable,
    b: &mut MatrixVariable,
    dest: &mut MatrixVariable,
) -> Result<(), Box<dyn std::error::Error>> {
    let dest_location = dest.default_location();

    // Only sync if variables are not already in the target location
    if a.default_location() != dest_location {
        a.sync_to(dest_location).await?;
    }
    if b.default_location() != dest_location {
        b.sync_to(dest_location).await?;
    }

    match dest_location {
        ResourceLocation::Local => {
            let a_matrix = a.read().await?;
            let b_matrix = b.read().await?;
            let result = a_matrix * b_matrix;
            dest.write(result).await?;
            dest.mark_updated(ResourceLocation::Local);
        }
        ResourceLocation::Memory => {
            let a_matrix = a.read().await?;
            let b_matrix = b.read().await?;
            let result = a_matrix * b_matrix;
            dest.write(result).await?;
            dest.mark_updated(ResourceLocation::Memory);
        }
        ResourceLocation::S3 => {
            let config = aws_config::load_from_env().await;
            let client = LambdaClient::new(&config);

            // Use the logged version with metadata that includes variable IDs
            call_matrix_multiplication_lambda_logged_with_metadata(
                &client,
                &a.s3_bucket(),
                &a.s3_key(),
                &b.s3_key(),
                &dest.s3_key(),
                a.id(),
                b.id(),
                dest.id(),
            )
            .await?;

            // Mark destination as updated in S3 since Lambda wrote to it
            dest.mark_updated(ResourceLocation::S3);
        }
    }
    Ok(())
}

/// Computes (A^T A + rho * I)^-1
pub async fn lasso_factor(
    a: &mut MatrixVariable,
    out: &mut MatrixVariable,
    rho: f32,
) -> Result<(), Box<dyn std::error::Error>> {
    let dest_location = out.default_location();

    if a.default_location() != dest_location {
        a.sync_to(dest_location).await?;
    }

    match dest_location {
        ResourceLocation::S3 => {
            let config = aws_config::load_from_env().await;
            let client = LambdaClient::new(&config);

            call_factor_lambda_logged(&client, &a.s3_bucket(), &a.s3_key(), &out.s3_key(), rho)
                .await?;

            out.mark_updated(ResourceLocation::S3);
        }
        ResourceLocation::Local => {
            let a_matrix = a.read().await?;
            let result = fast_lasso_inverse(&a_matrix, rho);
            out.write(result).await?;
            out.mark_updated(ResourceLocation::Local);
        }
        ResourceLocation::Memory => {
            let a_matrix = a.read().await?;
            let result = fast_lasso_inverse(&a_matrix, rho);
            out.write(result).await?;
            out.mark_updated(ResourceLocation::Memory);
        }
    }

    Ok(())
}

/// Soft thresholding operation for L1 regularization
/// Computes sign(x) * max(|x| - threshold, 0)
/// where threshold = lambda / (rho * n_subproblems)
pub async fn soft_threshold(
    x: &mut MatrixVariable,
    lambda: f32,
    rho: f32,
    n_subproblems: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let dest_location = x.default_location();

    // Always compute in memory for now, but only sync if needed
    if x.default_location() != ResourceLocation::Memory {
        x.sync_to(ResourceLocation::Memory).await?;
    }

    let x_matrix = x.read().await?;
    let threshold = lambda / (rho * n_subproblems as f32);

    // Compute sign(x)
    let x_sign = x_matrix.map(|v| {
        if v > 0.0 {
            1.0
        } else if v < 0.0 {
            -1.0
        } else {
            0.0
        }
    });

    // Compute sign(x) * max(|x| - threshold, 0)
    let result =
        x_sign.map_with_location(|i, j, sign| sign * (x_matrix[(i, j)].abs() - threshold).max(0.0));

    // Write result back to memory
    x.write(result).await?;

    // If the destination is S3, sync the result
    if dest_location == ResourceLocation::S3 {
        x.sync_to(ResourceLocation::S3).await?;
    }

    Ok(())
}

/// Scales a matrix by a given factor, in place.
pub async fn scale(
    matrix_var: &mut MatrixVariable,
    factor: f32,
) -> Result<(), Box<dyn std::error::Error>> {
    // No need to sync - just read from default location, compute, and write back
    let current_matrix = matrix_var.read().await?;
    let scaled_matrix = current_matrix * factor;
    matrix_var.write(scaled_matrix).await?;
    Ok(())
}
