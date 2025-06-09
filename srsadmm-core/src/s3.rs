use std::path::PathBuf;

use aws_sdk_s3::{
    Client as S3Client,
    primitives::{ByteStream, Length},
    types::{CompletedMultipartUpload, CompletedPart},
};
use std::fs::OpenOptions;
use std::io::Write;
use tokio::{
    fs::File,
    io::{AsyncBufRead, AsyncWriteExt},
};

pub(crate) const S3_CHUNK_SIZE: u64 = 1024 * 1024 * 5;
pub(crate) const S3_MAX_CHUNKS: u64 = 100000;

pub async fn download_file_to_local(
    bucket: &str,
    key: &str,
    local_path: &PathBuf,
) -> Result<(), Box<dyn std::error::Error>> {
    let start_time = std::time::Instant::now();
    let shared_config = aws_config::load_from_env().await;
    let client = S3Client::new(&shared_config);
    let mut file = File::create(local_path).await?;

    let get_object_res = client.get_object().bucket(bucket).key(key).send().await?;
    let mut stream = get_object_res.body;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        file.write_all(&chunk).await?;
    }
    log_s3_timing(
        key,
        "download_file",
        start_time.elapsed().as_millis() as f64,
    );
    Ok(())
}

// Uplaod a local file to S3 using multipart upload
// Lots taken from https://docs.aws.amazon.com/sdk-for-rust/latest/dg/rust_s3_code_examples.html
pub async fn upload_file_from_local(
    bucket: &str,
    key: &str,
    local_path: &PathBuf,
) -> Result<(), Box<dyn std::error::Error>> {
    let start_time = std::time::Instant::now();
    let shared_config = aws_config::load_from_env().await;
    let client = S3Client::new(&shared_config);

    let multipart_upload_res = client
        .create_multipart_upload()
        .bucket(bucket)
        .key(key)
        .send()
        .await?;

    let upload_id = multipart_upload_res
        .upload_id()
        .ok_or("Missing upload_id after CreateMultipartUpload")?;

    let file_size = tokio::fs::metadata(local_path)
        .await
        .expect("File to upload doesn't exist")
        .len();

    let mut chunk_count = (file_size / S3_CHUNK_SIZE) + 1;
    let mut size_of_last_chunk = file_size % S3_CHUNK_SIZE;
    if size_of_last_chunk == 0 {
        size_of_last_chunk = S3_CHUNK_SIZE;
        chunk_count -= 1;
    }

    if file_size == 0 {
        return Result::Err(format!("File {} is empty", key).into());
    }
    if chunk_count > S3_MAX_CHUNKS {
        return Result::Err(
            format!(
                "File {} has too many chunks. May require changing S3_CHUNK_SIZE or S3_MAX_CHUNKS",
                key
            )
            .into(),
        );
    }

    let mut upload_parts: Vec<aws_sdk_s3::types::CompletedPart> = Vec::new();

    for chunk_index in 0..chunk_count {
        let this_chunk = if chunk_count - 1 == chunk_index {
            size_of_last_chunk
        } else {
            S3_CHUNK_SIZE
        };
        let stream = ByteStream::read_from()
            .path(local_path)
            .offset(chunk_index * S3_CHUNK_SIZE)
            .length(Length::Exact(this_chunk))
            .build()
            .await
            .unwrap();

        // Chunk index needs to start at 0, but part numbers start at 1.
        let part_number = (chunk_index as i32) + 1;
        let upload_part_res = client
            .upload_part()
            .key(key)
            .bucket(bucket)
            .upload_id(upload_id)
            .body(stream)
            .part_number(part_number)
            .send()
            .await?;

        upload_parts.push(
            CompletedPart::builder()
                .e_tag(upload_part_res.e_tag.unwrap_or_default())
                .part_number(part_number)
                .build(),
        );
    }
    let completed_multipart_upload: CompletedMultipartUpload = CompletedMultipartUpload::builder()
        .set_parts(Some(upload_parts))
        .build();

    let _complete_multipart_upload_res = client
        .complete_multipart_upload()
        .bucket(bucket)
        .key(key)
        .multipart_upload(completed_multipart_upload)
        .upload_id(upload_id)
        .send()
        .await?;

    log_s3_timing(key, "upload_file", start_time.elapsed().as_millis() as f64);

    Ok(())
}

pub async fn download_file_to_vec(
    bucket: &str,
    key: &str,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let start_time = std::time::Instant::now();
    let shared_config = aws_config::load_from_env().await;
    let client = S3Client::new(&shared_config);
    let get_object_res = client.get_object().bucket(bucket).key(key).send().await?;
    let body = get_object_res.body.collect().await?;
    let bytes = body.to_vec();
    log_s3_timing(key, "download_vec", start_time.elapsed().as_millis() as f64);
    Ok(bytes)
}

pub async fn download_file_to_buf(
    bucket: &str,
    key: &str,
) -> Result<impl AsyncBufRead, Box<dyn std::error::Error>> {
    let shared_config = aws_config::load_from_env().await;
    let client = S3Client::new(&shared_config);
    let get_object_res = client.get_object().bucket(bucket).key(key).send().await?;
    return Ok(get_object_res.body.into_async_read());
}

// Uplaod a vector of bytes to S3 using multipart upload
// Lots taken from https://docs.aws.amazon.com/sdk-for-rust/latest/dg/rust_s3_code_examples.html
pub async fn upload_vec_to_s3(
    bucket: &str,
    key: &str,
    bytes: Vec<u8>,
) -> Result<(), Box<dyn std::error::Error>> {
    let start_time = std::time::Instant::now();
    let shared_config = aws_config::load_from_env().await;
    let client = S3Client::new(&shared_config);

    let multipart_upload_res = client
        .create_multipart_upload()
        .bucket(bucket)
        .key(key)
        .send()
        .await?;

    let upload_id = multipart_upload_res
        .upload_id()
        .ok_or("Missing upload_id after CreateMultipartUpload")?;

    let file_size = bytes.len() as u64;

    let mut chunk_count = (file_size / S3_CHUNK_SIZE) + 1;
    let mut size_of_last_chunk = file_size % S3_CHUNK_SIZE;
    if size_of_last_chunk == 0 {
        size_of_last_chunk = S3_CHUNK_SIZE;
        chunk_count -= 1;
    }

    if file_size == 0 {
        return Result::Err("Bytes are empty".into());
    }
    if chunk_count > S3_MAX_CHUNKS {
        return Result::Err(
            "Bytes are too large, may require changing S3_CHUNK_SIZE or S3_MAX_CHUNKS".into(),
        );
    }

    let mut upload_parts: Vec<aws_sdk_s3::types::CompletedPart> = Vec::new();

    for chunk_index in 0..chunk_count {
        let this_chunk = if chunk_count - 1 == chunk_index {
            size_of_last_chunk
        } else {
            S3_CHUNK_SIZE
        };

        let chunk = bytes[(chunk_index * S3_CHUNK_SIZE) as usize
            ..(chunk_index * S3_CHUNK_SIZE + this_chunk) as usize]
            .to_vec();
        let stream = ByteStream::from(chunk);

        // Chunk index needs to start at 0, but part numbers start at 1.
        let part_number = (chunk_index as i32) + 1;
        let upload_part_res = client
            .upload_part()
            .key(key)
            .bucket(bucket)
            .upload_id(upload_id)
            .body(stream)
            .part_number(part_number)
            .send()
            .await?;

        upload_parts.push(
            CompletedPart::builder()
                .e_tag(upload_part_res.e_tag.unwrap_or_default())
                .part_number(part_number)
                .build(),
        );
    }
    let completed_multipart_upload: CompletedMultipartUpload = CompletedMultipartUpload::builder()
        .set_parts(Some(upload_parts))
        .build();

    let _complete_multipart_upload_res = client
        .complete_multipart_upload()
        .bucket(bucket)
        .key(key)
        .multipart_upload(completed_multipart_upload)
        .upload_id(upload_id)
        .send()
        .await?;

    log_s3_timing(key, "upload_vec", start_time.elapsed().as_millis() as f64);

    Ok(())
}

pub async fn delete_s3_object(bucket: &str, key: &str) -> Result<(), Box<dyn std::error::Error>> {
    let shared_config = aws_config::load_from_env().await;
    let client = S3Client::new(&shared_config);
    client
        .delete_object()
        .bucket(bucket)
        .key(key)
        .send()
        .await?;
    Ok(())
}

pub(crate) fn log_s3_timing(id: &str, operation: &str, duration_ms: f64) {
    let log_entry = format!(
        "{},{},{:.3},{}\n",
        id,
        duration_ms,
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs(),
        operation
    );

    // Try to append to lambda_timings.csv, create if it doesn't exist
    if let Ok(mut file) = OpenOptions::new()
        .create(true)
        .append(true)
        .open("s3_timings.csv")
    {
        // Write header if file is new (check file size)
        if file.metadata().map(|m| m.len()).unwrap_or(1) == 0 {
            let _ = writeln!(file, "file_id,duration_ms,timestamp,operation");
        }
        let _ = write!(file, "{}", log_entry);
    }
}
