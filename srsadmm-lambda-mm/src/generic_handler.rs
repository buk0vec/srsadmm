use async_compression::tokio::bufread::{ZstdDecoder, ZstdEncoder};
use tokio::io::{AsyncReadExt, BufReader};

use lambda_runtime::{Error, LambdaEvent};
use serde::{Deserialize, Serialize};
use aws_sdk_s3 as s3;
extern crate nalgebra as na;
use bincode::config::Configuration;
use thiserror::Error;

const BINCODE_CONFIG: Configuration = bincode::config::standard()
    .with_little_endian()
    .with_variable_int_encoding();

// TODO: implement better error handling
#[derive(Error, Debug)]
pub enum LambdaError {
    #[error("Matrix dimensions {0}x{1} and {2}x{3} do not match")]
    DimMismatchError(usize, usize, usize, usize),
    #[error("Bincode encode error: {0}")]
    BincodeEncodeError(#[from] bincode::error::EncodeError),
    #[error("Bincode decode error: {0}")]
    BincodeDecodeError(#[from] bincode::error::DecodeError),
}

#[derive(Deserialize)]
pub(crate) struct IncomingMessage {
    bucket: String,
    matrix1_key: String,
    matrix2_key: String,
    out_key: String,
}

#[derive(Serialize)]
pub(crate) struct OutgoingMessage {
    status: String
}

/// This is the main body for the function.
/// Write your code inside it.
/// There are some code example in the following URLs:
/// - https://github.com/awslabs/aws-lambda-rust-runtime/tree/main/examples
/// - https://github.com/aws-samples/serverless-rust-demo/
pub(crate) async fn function_handler(event: LambdaEvent<IncomingMessage>) -> Result<OutgoingMessage, Error> {
    // Extract some useful info from the request
    println!("Bucket: {}", event.payload.bucket);
    println!("Matrix1 Key: {}", event.payload.matrix1_key);
    println!("Matrix2 Key: {}", event.payload.matrix2_key);
    println!("Out Key: {}", event.payload.out_key);

    let bucket = event.payload.bucket;
    let matrix1_key = event.payload.matrix1_key;
    let matrix2_key = event.payload.matrix2_key;
    let out_key = event.payload.out_key;

    let config = aws_config::load_from_env().await;
    let client = s3::Client::new(&config);


    println!("Getting matrix1 from {}", matrix1_key);
    let response = client.get_object().bucket(&bucket).key(&matrix1_key).send().await?;
    let bytes = response.body.into_async_read();

    // let (matrix1, _): (na::DMatrix<f32>, _) = bincode::serde::decode_from_reader(&mut reader, BINCODE_CONFIG)?;

    let mut decoder = ZstdDecoder::new(bytes);
    println!("Decoding matrix1");
    let mut decompressed = Vec::new();
    let mut buf = [0; 1024];
    loop {
        let n = decoder.read(&mut buf).await?;
        if n == 0 {
            break;
        }
        decompressed.extend_from_slice(&buf[..n]);
    }
    println!("Recieved {} bytes", decompressed.len());
    let (matrix1, _): (na::DMatrix<f32>, _) = bincode::serde::decode_from_slice(&decompressed, BINCODE_CONFIG)?;

    println!("Getting matrix2 from {}", matrix2_key);
    let response = client.get_object().bucket(&bucket).key(&matrix2_key).send().await?;
    let bytes = response.body.into_async_read();

    let mut decoder = ZstdDecoder::new(bytes);
    let mut decompressed = Vec::new();
    let mut buf = [0; 1024];
    loop {
        let n = decoder.read(&mut buf).await?;
        if n == 0 {
            break;
        }
        decompressed.extend_from_slice(&buf[..n]);
    }
    println!("Recieved {} bytes", decompressed.len());
    let (matrix2, _): (na::DMatrix<f32>, _) = bincode::serde::decode_from_slice(&decompressed, BINCODE_CONFIG)?;

    println!("Matrix1: {}x{}", matrix1.nrows(), matrix1.ncols());
    println!("Matrix2: {}x{}", matrix2.nrows(), matrix2.ncols());
    println!("Multiplying matrices");
    
    // Make sure matrices can be multiplied
    if matrix1.ncols() != matrix2.nrows() {
        return Err(LambdaError::DimMismatchError(matrix1.nrows(), matrix1.ncols(), matrix2.nrows(), matrix2.ncols()).into());
    }
    
    let result = matrix1 * matrix2;

    let encoded = bincode::serde::encode_to_vec(&result, BINCODE_CONFIG).map_err(LambdaError::BincodeEncodeError)?;

    let mut encoder = ZstdEncoder::with_quality(&encoded[..], async_compression::Level::Precise(3));
    let mut buf = [0; 1024];
    let mut compressed = Vec::new();
    loop {
        let n = encoder.read(&mut buf).await?;
        if n == 0 {
            break;
        }
        compressed.extend_from_slice(&buf[..n]);
    }

    let body_length = compressed.len() as i64;
    println!("Putting result");
    client.put_object()
        .bucket(&bucket)
        .key(&out_key)
        .body(compressed.into())
        .content_length(body_length)
        .send()
        .await?;

    // Return `OutgoingMessage` (it will be serialized to JSON automatically by the runtime)
    Ok(OutgoingMessage {
        status: "Success".to_string()
    })
}


// cargo lambda invoke --data-ascii "{\"bucket\": \"srsadmm\", \"matrix1_key\":\"subproblem_9_matrix_at_a_rho_i_inv.bin\",\"matrix2_key\":\"subproblem_9_matrix_rho_z_minus_y_minus_at_b.bin\",\"out_key\":\"test_mm.bin\"}"
// #[cfg(test)]
// mod tests {
//     use super::*;
//     use lambda_runtime::{Context, LambdaEvent};

//     #[tokio::test]
//     async fn test_generic_handler() {
//         let event = LambdaEvent::new(IncomingMessage { command: "test".to_string() }, Context::default());
//         let response = function_handler(event).await.unwrap();
//         assert_eq!(response.msg, "Command test.");
//     }
// }

// cargo lambda invoke --data-ascii "{\"bucket\": \"srsadmm\", \"matrix1_key\":\"lassosubproblem-8-at_a_rho_i_inv\",\"matrix2_key\":\"lassosubproblem-8-atb-rho-z-y\",\"out_key\":\"test_mm_encoded\"}"
// cargo lambda invoke --data-ascii "{\"bucket\":\"srsadmm\",\"matrix1_key\":\"lassosubproblem-8-at_a_rho_i_inv\",\"matrix2_key\":\"lassosubproblem-8-atb-rho-z-y\",\"out_key\":\"lassosubproblem-8-x\"}"