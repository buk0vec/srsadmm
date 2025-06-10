use crate::{timing::TimingTracker, utils::LassoError};
use aws_sdk_lambda::Client as LambdaClient;
use aws_smithy_types::Blob;
use serde::{Deserialize, Serialize};
use std::{error::Error, fs::OpenOptions, io::Write, time::Instant};
use tokio::time::{sleep, Duration};

/// Payload for matrix multiplication Lambda function calls.
///
/// This structure contains the S3 object keys and bucket information
/// needed to perform distributed matrix multiplication via AWS Lambda.
#[derive(Serialize)]
pub struct MatrixMultiplicationPayload {
    /// S3 bucket containing the matrices
    pub bucket: String,
    /// S3 key for the first matrix operand
    pub matrix1_key: String,
    /// S3 key for the second matrix operand
    pub matrix2_key: String,
    /// S3 key where the result matrix should be stored
    pub out_key: String,
}

/// Payload for matrix factorization Lambda function calls.
///
/// This structure contains the information needed to compute
/// matrix factorizations required in ADMM algorithms.
#[derive(Serialize)]
pub struct FactorPayload {
    /// S3 bucket containing the matrix
    pub bucket: String,
    /// S3 key for the matrix to factorize
    pub a_key: String,
    /// S3 key where the factorization result should be stored
    pub out_key: String,
    /// Regularization parameter rho
    pub rho: f32,
}

/// Response structure from Lambda matrix operations.
///
/// All Lambda functions return this standardized response format
/// to indicate success or failure of the requested operation.
#[derive(Deserialize)]
pub struct MatrixOpResponse {
    /// Status of the operation ("success" or error message)
    pub status: String,
}

/// Calls the matrix multiplication Lambda function to multiply two matrices stored in S3
///
/// # Arguments
/// * `client` - The AWS Lambda client
/// * `bucket` - The S3 bucket name
/// * `matrix1_key` - The S3 key for the first matrix
/// * `matrix2_key` - The S3 key for the second matrix
/// * `out_key` - The S3 key where the result should be stored
///
/// # Returns
/// * `Result<(), Box<dyn Error>>` - Ok(()) if successful, error otherwise
pub async fn call_matrix_multiplication_lambda(
    client: &LambdaClient,
    bucket: &str,
    matrix1_key: &str,
    matrix2_key: &str,
    out_key: &str,
) -> Result<(), Box<dyn Error>> {
    const MAX_RETRIES: u32 = 3;
    let mut last_error = None;

    for attempt in 0..=MAX_RETRIES {
        let payload = MatrixMultiplicationPayload {
            bucket: bucket.to_string(),
            matrix1_key: matrix1_key.to_string(),
            matrix2_key: matrix2_key.to_string(),
            out_key: out_key.to_string(),
        };

        let payload = serde_json::to_string(&payload)?;
        
        match client
            .invoke()
            .function_name("srsadmm-lambda-mm")
            .payload(Blob::new(payload))
            .send()
            .await
        {
            Ok(response) => {
                let payload = response.payload().ok_or("Failed to get payload")?;
                let response_str = match std::str::from_utf8(payload.as_ref()) {
                    Ok(s) => s,
                    Err(e) => {
                        println!("[Lambda] Error converting payload to string: {}", e);
                        return Err(e.into());
                    }
                };
                
                match serde_json::from_str::<MatrixOpResponse>(response_str) {
                    Ok(matrix_response) => {
                        if matrix_response.status.to_lowercase() != "success" {
                            let error = format!("Lambda function failed: {}", matrix_response.status);
                            last_error = Some(error.clone());
                            
                            if attempt < MAX_RETRIES {
                                println!("[Lambda] Attempt {} failed: {}. Retrying...", attempt + 1, error);
                                // Exponential backoff: 1s, 2s, 4s
                                let delay = Duration::from_millis(1000 * (1 << attempt));
                                sleep(delay).await;
                                continue;
                            } else {
                                return Err(error.into());
                            }
                        } else {
                            // Success
                            if attempt > 0 {
                                println!("[Lambda] Matrix multiplication succeeded on attempt {}", attempt + 1);
                            }
                            return Ok(());
                        }
                    }
                    Err(e) => {
                        let error = format!("Failed to parse response: {}", e);
                        last_error = Some(error.clone());
                        
                        if attempt < MAX_RETRIES {
                            println!("[Lambda] Attempt {} failed: {}. Retrying...", attempt + 1, error);
                            let delay = Duration::from_millis(1000 * (1 << attempt));
                            sleep(delay).await;
                            continue;
                        } else {
                            return Err(error.into());
                        }
                    }
                }
            }
            Err(e) => {
                let error = format!("Lambda invocation failed: {}", e);
                last_error = Some(error.clone());
                
                if attempt < MAX_RETRIES {
                    println!("[Lambda] Attempt {} failed: {}. Retrying...", attempt + 1, error);
                    // Exponential backoff: 1s, 2s, 4s
                    let delay = Duration::from_millis(1000 * (1 << attempt));
                    sleep(delay).await;
                    continue;
                } else {
                    return Err(e.into());
                }
            }
        }
    }

    // This should never be reached, but just in case
    Err(last_error.unwrap_or_else(|| "Unknown error".to_string()).into())
}

/// Calls the matrix multiplication Lambda function with timing tracking
///
/// # Arguments
/// * `client` - The AWS Lambda client
/// * `tracker` - The timing tracker to record execution time
/// * `bucket` - The S3 bucket name
/// * `matrix1_key` - The S3 key for the first matrix
/// * `matrix2_key` - The S3 key for the second matrix
/// * `out_key` - The S3 key where the result should be stored
///
/// # Returns
/// * `Result<(), LassoError>` - Ok(()) if successful, error otherwise
pub async fn call_matrix_multiplication_lambda_timed(
    client: &LambdaClient,
    tracker: &mut TimingTracker,
    bucket: &str,
    matrix1_key: &str,
    matrix2_key: &str,
    out_key: &str,
) -> Result<(), LassoError> {
    let start = Instant::now();

    let result =
        call_matrix_multiplication_lambda(client, bucket, matrix1_key, matrix2_key, out_key).await;

    let duration = start.elapsed();
    tracker.record_lambda("srsadmm-lambda-gemm", "matrix_multiplication", duration);

    result
        .map_err(|e| LassoError::from_string(format!("Matrix multiplication lambda failed: {}", e)))
}

/// Logs lambda timing to a CSV file
pub(crate) fn log_lambda_timing(
    function_name: &str,
    operation_type: &str,
    duration_ms: f64,
    metadata: Option<&str>,
) {
    let log_entry = format!(
        "{},{},{:.3},{},{}\n",
        function_name,
        operation_type,
        duration_ms,
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs(),
        metadata.unwrap_or("")
    );

    // Try to append to lambda_timings.csv, create if it doesn't exist
    if let Ok(mut file) = OpenOptions::new()
        .create(true)
        .append(true)
        .open("lambda_timings.csv")
    {
        // Write header if file is new (check file size)
        if file.metadata().map(|m| m.len()).unwrap_or(1) == 0 {
            let _ = writeln!(
                file,
                "function_name,operation_type,duration_ms,timestamp,metadata"
            );
        }
        let _ = write!(file, "{}", log_entry);
    }
}

/// Calls the matrix multiplication Lambda function with automatic timing logging
pub async fn call_matrix_multiplication_lambda_logged(
    client: &LambdaClient,
    bucket: &str,
    matrix1_key: &str,
    matrix2_key: &str,
    out_key: &str,
) -> Result<(), LassoError> {
    let start = Instant::now();

    let result =
        call_matrix_multiplication_lambda(client, bucket, matrix1_key, matrix2_key, out_key).await;

    let duration = start.elapsed();
    log_lambda_timing(
        "srsadmm-lambda-gemm",
        "matrix_multiplication",
        duration.as_secs_f64() * 1000.0,
        None,
    );

    result
        .map_err(|e| LassoError::from_string(format!("Matrix multiplication lambda failed: {}", e)))
}

/// Calls the matrix multiplication Lambda function with automatic timing logging and metadata
pub async fn call_matrix_multiplication_lambda_logged_with_metadata(
    client: &LambdaClient,
    bucket: &str,
    matrix1_key: &str,
    matrix2_key: &str,
    out_key: &str,
    matrix1_id: &str,
    matrix2_id: &str,
    out_id: &str,
) -> Result<(), LassoError> {
    let start = Instant::now();

    let result =
        call_matrix_multiplication_lambda(client, bucket, matrix1_key, matrix2_key, out_key).await;

    let duration = start.elapsed();
    let metadata = format!(
        "a_id:{}::b_id:{}::dest_id:{}",
        matrix1_id, matrix2_id, out_id
    );
    log_lambda_timing(
        "srsadmm-lambda-gemm",
        "matrix_multiplication",
        duration.as_secs_f64() * 1000.0,
        Some(&metadata),
    );

    result
        .map_err(|e| LassoError::from_string(format!("Matrix multiplication lambda failed: {}", e)))
}

pub async fn call_factor_lambda(
    client: &LambdaClient,
    bucket: &str,
    a_key: &str,
    out_key: &str,
    rho: f32,
) -> Result<(), Box<dyn Error>> {
    const MAX_RETRIES: u32 = 3;
    let mut last_error = None;

    for attempt in 0..=MAX_RETRIES {
        let payload = FactorPayload {
            bucket: bucket.to_string(),
            a_key: a_key.to_string(),
            out_key: out_key.to_string(),
            rho: rho,
        };

        let payload = serde_json::to_string(&payload)?;
        
        match client
            .invoke()
            .function_name("srsadmm-lasso-factor")
            .payload(Blob::new(payload))
            .send()
            .await
        {
            Ok(response) => {
                let payload = response.payload().ok_or("Failed to get payload")?;
                
                match std::str::from_utf8(payload.as_ref()) {
                    Ok(response_str) => {
                        match serde_json::from_str::<MatrixOpResponse>(response_str) {
                            Ok(matrix_response) => {
                                if matrix_response.status.to_lowercase() != "success" {
                                    let error = format!("Lambda function failed: {}", matrix_response.status);
                                    last_error = Some(error.clone());
                                    
                                    if attempt < MAX_RETRIES {
                                        println!("[Lambda] Attempt {} failed: {}. Retrying...", attempt + 1, error);
                                        // Exponential backoff: 1s, 2s, 4s
                                        let delay = Duration::from_millis(1000 * (1 << attempt));
                                        sleep(delay).await;
                                        continue;
                                    } else {
                                        return Err(error.into());
                                    }
                                } else {
                                    // Success
                                    if attempt > 0 {
                                        println!("[Lambda] Factor operation succeeded on attempt {}", attempt + 1);
                                    }
                                    return Ok(());
                                }
                            }
                            Err(e) => {
                                let error = format!("Failed to parse response: {}", e);
                                last_error = Some(error.clone());
                                
                                if attempt < MAX_RETRIES {
                                    println!("[Lambda] Attempt {} failed: {}. Retrying...", attempt + 1, error);
                                    let delay = Duration::from_millis(1000 * (1 << attempt));
                                    sleep(delay).await;
                                    continue;
                                } else {
                                    return Err(error.into());
                                }
                            }
                        }
                    }
                    Err(e) => {
                        let error = format!("Failed to convert response to string: {}", e);
                        last_error = Some(error.clone());
                        
                        if attempt < MAX_RETRIES {
                            println!("[Lambda] Attempt {} failed: {}. Retrying...", attempt + 1, error);
                            let delay = Duration::from_millis(1000 * (1 << attempt));
                            sleep(delay).await;
                            continue;
                        } else {
                            return Err(error.into());
                        }
                    }
                }
            }
            Err(e) => {
                let error = format!("Lambda invocation failed: {}", e);
                last_error = Some(error.clone());
                
                if attempt < MAX_RETRIES {
                    println!("[Lambda] Attempt {} failed: {}. Retrying...", attempt + 1, error);
                    // Exponential backoff: 1s, 2s, 4s
                    let delay = Duration::from_millis(1000 * (1 << attempt));
                    sleep(delay).await;
                    continue;
                } else {
                    return Err(e.into());
                }
            }
        }
    }

    // This should never be reached, but just in case
    Err(last_error.unwrap_or_else(|| "Unknown error".to_string()).into())
}

pub async fn call_factor_lambda_logged(
    client: &LambdaClient,
    bucket: &str,
    a_key: &str,
    out_key: &str,
    rho: f32,
) -> Result<(), LassoError> {
    let start = Instant::now();

    let result = call_factor_lambda(client, bucket, a_key, out_key, rho).await;

    let duration = start.elapsed();
    log_lambda_timing(
        "srsadmm-lambda-factor",
        "factor",
        duration.as_secs_f64() * 1000.0,
        None,
    );

    result.map_err(|e| LassoError::from_string(format!("Factor lambda failed: {}", e)))
}
