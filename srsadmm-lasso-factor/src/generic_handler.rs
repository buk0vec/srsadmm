use lambda_runtime::{Error, LambdaEvent};
use serde::{Deserialize, Serialize};
use aws_sdk_s3 as s3;
use tokio::io::{AsyncReadExt};

extern crate nalgebra as na;
use bincode::config::Configuration;
use async_compression::tokio::bufread::{ZstdDecoder, ZstdEncoder};

const BINCODE_CONFIG: Configuration = bincode::config::standard()
    .with_little_endian()
    .with_variable_int_encoding();


/// This is a made-up example. Incoming messages come into the runtime as unicode
/// strings in json format, which can map to any structure that implements `serde::Deserialize`
/// The runtime pays no attention to the contents of the incoming message payload.
#[derive(Deserialize)]
pub(crate) struct IncomingMessage {
    // S3 bucket name
    bucket: String,
    // Location of matrix A in S3
    a_key: String,
    // Output location in S3
    out_key: String,
    // Rho
    rho: f32
}

/// This is a made-up example of what an outgoing message structure may look like.
/// There is no restriction on what it can be. The runtime requires responses
/// to be serialized into json. The runtime pays no attention
/// to the contents of the outgoing message payload.
#[derive(Serialize)]
pub(crate) struct OutgoingMessage {
    status: String,
}

/// This is the main body for the function.
/// Write your code inside it.
/// There are some code example in the following URLs:
/// - https://github.com/awslabs/aws-lambda-rust-runtime/tree/main/examples
/// - https://github.com/aws-samples/serverless-rust-demo/
pub(crate) async fn function_handler(event: LambdaEvent<IncomingMessage>) -> Result<OutgoingMessage, Error> {
    // Extract some useful info from the request
    let bucket = event.payload.bucket;
    let a_key = event.payload.a_key;
    let out_key = event.payload.out_key;
    let rho = event.payload.rho;

    let config = aws_config::from_env().load().await;
    let client = s3::Client::new(&config);


    let response = client.get_object().bucket(&bucket).key(&a_key).send().await?;
    let bytes = response.body.into_async_read();

    let mut decoder = ZstdDecoder::new(bytes);
    let mut decompressed = Vec::new();
    let mut buffer = [0; 1024 * 1024];

    loop {
        let n = decoder.read(&mut buffer).await?;
        if n == 0 {
            break;
        }
        decompressed.extend_from_slice(&buffer[..n]);
    }

    let (a, _): (na::DMatrix<f32>, _) = bincode::serde::decode_from_slice(&decompressed, BINCODE_CONFIG)?;

    let a_inv = fast_lasso_inverse(&a, rho);

    let encoded = bincode::serde::encode_to_vec(&a_inv, BINCODE_CONFIG)?;

    let mut encoder = ZstdEncoder::with_quality(&encoded[..], async_compression::Level::Precise(3));
    let mut compressed = Vec::new();
    let mut buffer = [0; 1024 * 1024];

    loop {
        let n = encoder.read(&mut buffer).await?;
        if n == 0 {
            break;
        }
        compressed.extend_from_slice(&buffer[..n]);
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

/// Helper function to quickly compute (A^T A + rho I)^-1
pub fn fast_lasso_inverse(a: &na::DMatrix<f32>, rho: f32) -> na::DMatrix<f32> {
    let m = a.nrows();
    let n = a.ncols();

    let eye_n = na::DMatrix::<f32>::identity(n, n);

    return if m < n / 2 {
        // For "fat" matrices, use Sherman-Morrison-Woodbury formula
        let rho_inv = 1.0 / rho;
        let eye_m = na::DMatrix::<f32>::identity(m, m);

        // I + A*A^T/rho
        let i_aat = a * a.transpose() * rho_inv + &eye_m;
        let i_aat_inv = i_aat
            .try_inverse()
            .ok_or(format!("Fast matrix inversion failed"))
            .unwrap();

        // (A^TA + rho*I)^-1 = rho^-1*I - rho^-2*A^T*(I+AA^T/rho)^-1*A
        eye_n * rho_inv - (a.transpose() * i_aat_inv * a) * (rho_inv * rho_inv)
    } else {
        // Cholesky decomposition
        let mut ata = a.transpose() * a;
        ata = ata + eye_n * rho;
        let l = ata.cholesky().expect("Failed to compute Cholesky decomposition");
        l.inverse()
    };
}
