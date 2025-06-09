use crate::{
    resource::{BINCODE_CONFIG, LocalConfig, MemoryConfig, ResourceLocation, S3Config},
    s3::{
        delete_s3_object, download_file_to_buf, download_file_to_local, upload_file_from_local,
        upload_vec_to_s3,
    },
};
use async_compression::tokio::bufread::ZstdDecoder;
use async_compression::tokio::bufread::ZstdEncoder;
use async_trait::async_trait;
use serde::Serialize;
use serde::de::DeserializeOwned;
use std::{
    marker::PhantomData,
    path::{Path, PathBuf},
    sync::RwLock,
};
use tokio::{
    fs::{self},
    io::AsyncReadExt,
};

#[async_trait]
pub(crate) trait StorageBackend<T: Serialize + DeserializeOwned + Clone + Send + Sync>:
    Send + Sync
{
    async fn read(&self, id: &str) -> Result<T, Box<dyn std::error::Error>>;
    async fn write(&self, id: &str, value: T) -> Result<(), Box<dyn std::error::Error>>;
    async fn delete(&self, id: &str) -> Result<(), Box<dyn std::error::Error>>;
    async fn copy_from(
        &self,
        id: &str,
        src: &(dyn StorageBackend<T> + Send + Sync),
    ) -> Result<(), Box<dyn std::error::Error>>;
    fn location(&self) -> ResourceLocation;
    fn bucket(&self) -> String;
    fn key(&self, id: &str) -> String;
    fn file_path(&self, id: &str) -> PathBuf;
}

/// Storage backend for storing values in memory. Values are cloned when read.
pub(crate) struct MemoryStorageBackend<T: Serialize + DeserializeOwned + Clone + Send + Sync> {
    #[allow(dead_code)]
    config: MemoryConfig,
    value: RwLock<Option<T>>,
}

impl<T> MemoryStorageBackend<T>
where
    T: Serialize + DeserializeOwned + Clone + Send + Sync,
{
    pub fn new(config: &MemoryConfig) -> Self {
        MemoryStorageBackend {
            config: config.clone(),
            value: RwLock::new(None),
        }
    }
}

#[async_trait]
impl<T> StorageBackend<T> for MemoryStorageBackend<T>
where
    T: Serialize + DeserializeOwned + Clone + Send + Sync,
{
    async fn read(&self, id: &str) -> Result<T, Box<dyn std::error::Error>> {
        match &*self.value.read().unwrap() {
            Some(value) => Ok(value.clone()),
            None => Err(format!("Failed to read value for id {}", id).into()),
        }
    }

    async fn write(&self, _id: &str, value: T) -> Result<(), Box<dyn std::error::Error>> {
        *self.value.write().unwrap() = Some(value);
        Ok(())
    }

    async fn delete(&self, _id: &str) -> Result<(), Box<dyn std::error::Error>> {
        *self.value.write().unwrap() = None;
        Ok(())
    }

    async fn copy_from(
        &self,
        id: &str,
        src: &(dyn StorageBackend<T> + Send + Sync),
    ) -> Result<(), Box<dyn std::error::Error>> {
        let value = src.read(id).await?;
        self.write(id, value).await?;
        Ok(())
    }

    fn location(&self) -> ResourceLocation {
        ResourceLocation::Memory
    }

    fn bucket(&self) -> String {
        String::new()
    }

    fn key(&self, _id: &str) -> String {
        String::new()
    }

    fn file_path(&self, _id: &str) -> PathBuf {
        PathBuf::new()
    }
}

/// Storage backend for storing values on disk.
pub(crate) struct LocalStorageBackend<T> {
    config: LocalConfig,
    _marker: PhantomData<T>,
}

impl<T> LocalStorageBackend<T> {
    pub fn new(config: &LocalConfig) -> Self {
        LocalStorageBackend {
            config: config.clone(),
            _marker: PhantomData,
        }
    }
}

#[async_trait]
impl<T> StorageBackend<T> for LocalStorageBackend<T>
where
    T: Serialize + DeserializeOwned + Clone + Send + Sync,
{
    async fn read(&self, id: &str) -> Result<T, Box<dyn std::error::Error>> {
        let path = self.file_path(id);
        let bytes = fs::read(path).await?;
        let mut decoder = ZstdDecoder::new(&bytes[..]);
        let mut decompressed = Vec::new();
        let mut buf = [0; 1024];
        loop {
            let n = decoder.read(&mut buf).await?;
            if n == 0 {
                break;
            }
            decompressed.extend_from_slice(&buf[..n]);
        }
        let (value, _): (T, _) = bincode::serde::decode_from_slice(&decompressed, BINCODE_CONFIG)?;
        Ok(value)
    }

    async fn write(&self, id: &str, value: T) -> Result<(), Box<dyn std::error::Error>> {
        let path = self.file_path(id);
        let bytes = bincode::serde::encode_to_vec(&value, BINCODE_CONFIG)?;
        let mut encoder =
            ZstdEncoder::with_quality(&bytes[..], async_compression::Level::Precise(3));
        let mut buf = [0; 1024];
        let mut compressed = Vec::new();
        loop {
            let n = encoder.read(&mut buf).await?;
            if n == 0 {
                break;
            }
            compressed.extend_from_slice(&buf[..n]);
        }
        fs::write(path, compressed).await?;
        Ok(())
    }

    async fn delete(&self, id: &str) -> Result<(), Box<dyn std::error::Error>> {
        let path = self.file_path(id);
        fs::remove_file(path).await?;
        Ok(())
    }

    async fn copy_from(
        &self,
        id: &str,
        src: &(dyn StorageBackend<T> + Send + Sync),
    ) -> Result<(), Box<dyn std::error::Error>> {
        match src.location() {
            ResourceLocation::S3 => {
                // Optimization: download file directly
                let bucket = src.bucket();
                let key = src.key(id);
                let dest_path = self.file_path(id);
                download_file_to_local(&bucket, &key, &dest_path).await?;
                Ok(())
            }
            _ => {
                let value = src.read(id).await?;
                self.write(id, value).await?;
                Ok(())
            }
        }
    }

    fn location(&self) -> ResourceLocation {
        ResourceLocation::Local
    }

    fn bucket(&self) -> String {
        String::new()
    }

    fn key(&self, _id: &str) -> String {
        String::new()
    }

    fn file_path(&self, id: &str) -> PathBuf {
        Path::new(&self.config.root)
            .join(format!("{}{}", self.config.prefix, id))
            .with_extension("bin.zst")
    }
}

/// Storage backend for storing values on S3.
pub(crate) struct S3StorageBackend<T> {
    config: S3Config,
    _marker: PhantomData<T>,
}

impl<T> S3StorageBackend<T> {
    pub fn new(config: &S3Config) -> Self {
        S3StorageBackend {
            config: config.clone(),
            _marker: PhantomData,
        }
    }
}

#[async_trait]
impl<T> StorageBackend<T> for S3StorageBackend<T>
where
    T: Serialize + DeserializeOwned + Clone + Send + Sync,
{
    async fn read(&self, id: &str) -> Result<T, Box<dyn std::error::Error>> {
        let bucket = self.bucket();
        let key = self.key(id);
        let mut stream = download_file_to_buf(&bucket, &key).await?;
        let mut decoder = ZstdDecoder::new(&mut stream);
        let mut decompressed = Vec::new();
        let mut buf = [0; 1024];
        loop {
            let n = decoder.read(&mut buf).await?;
            if n == 0 {
                break;
            }
            decompressed.extend_from_slice(&buf[..n]);
        }
        let (value, _): (T, _) = bincode::serde::decode_from_slice(&decompressed, BINCODE_CONFIG)?;
        Ok(value)
    }

    async fn write(&self, id: &str, value: T) -> Result<(), Box<dyn std::error::Error>> {
        let bucket = self.bucket();
        let key = self.key(id);
        let bytes = bincode::serde::encode_to_vec(&value, BINCODE_CONFIG)?;
        let mut encoder =
            ZstdEncoder::with_quality(&bytes[..], async_compression::Level::Precise(3));
        let mut buf = [0; 1024];
        let mut compressed = Vec::new();
        loop {
            let n = encoder.read(&mut buf).await?;
            if n == 0 {
                break;
            }
            compressed.extend_from_slice(&buf[..n]);
        }
        upload_vec_to_s3(&bucket, &key, compressed).await?;
        Ok(())
    }

    async fn delete(&self, id: &str) -> Result<(), Box<dyn std::error::Error>> {
        let bucket = self.bucket();
        let key = self.key(id);
        delete_s3_object(&bucket, &key).await?;
        Ok(())
    }

    async fn copy_from(
        &self,
        id: &str,
        src: &(dyn StorageBackend<T> + Send + Sync),
    ) -> Result<(), Box<dyn std::error::Error>> {
        match src.location() {
            ResourceLocation::Local => {
                let src_path = src.file_path(id);
                let bucket = self.bucket();
                let key = self.key(id);
                upload_file_from_local(&bucket, &key, &src_path).await?;
                Ok(())
            }
            _ => {
                let value = src.read(id).await?;
                self.write(id, value).await?;
                Ok(())
            }
        }
    }

    fn location(&self) -> ResourceLocation {
        ResourceLocation::S3
    }

    fn bucket(&self) -> String {
        self.config.bucket.clone()
    }

    fn key(&self, id: &str) -> String {
        format!("{}{}", self.config.prefix, id)
    }

    fn file_path(&self, _id: &str) -> PathBuf {
        PathBuf::new()
    }
}
