use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::Arc;
use std::{marker::PhantomData, path::Path};

use serde::{Deserialize, Serialize, de::DeserializeOwned};

use crate::storage::{LocalStorageBackend, MemoryStorageBackend, S3StorageBackend, StorageBackend};

// TODO: standardize across codebase
pub(crate) const BINCODE_CONFIG: bincode::config::Configuration = bincode::config::standard()
    .with_little_endian()
    .with_variable_int_encoding();

/// Configuration for Amazon S3 storage backend.
///
/// This configuration specifies the S3 bucket, region, and key prefix
/// for storing ADMM problem data in the cloud.
///
/// # Example
///
/// ```rust
/// # use srsadmm_core::resource::S3Config;
///
/// let config = S3Config::new("my-admm-bucket", "us-west-2", "experiments/");
/// ```
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct S3Config {
    /// The S3 bucket name where data will be stored
    pub bucket: String,
    /// The AWS region where the bucket is located
    pub region: String,
    /// Key prefix for organizing objects within the bucket
    pub prefix: String,
}

impl S3Config {
    /// Creates a new S3 configuration.
    ///
    /// # Arguments
    ///
    /// * `bucket` - The S3 bucket name
    /// * `region` - The AWS region (e.g., "us-west-2")
    /// * `prefix` - Key prefix for organizing objects
    ///
    /// # Returns
    ///
    /// A new `S3Config` instance
    pub fn new(bucket: &str, region: &str, prefix: &str) -> Self {
        S3Config {
            bucket: bucket.to_string(),
            region: region.to_string(),
            prefix: prefix.to_string(),
        }
    }
}

/// Configuration for local filesystem storage backend.
///
/// This configuration specifies the root directory and file prefix
/// for storing ADMM problem data on the local filesystem.
///
/// # Example
///
/// ```rust
/// # use srsadmm_core::resource::LocalConfig;
/// # use std::path::Path;
///
/// let config = LocalConfig::new(Path::new("/tmp/admm"), "exp1_");
/// ```
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LocalConfig {
    /// Root directory path for storing files
    pub root: String,
    /// Filename prefix for organizing files
    pub prefix: String,
}

impl LocalConfig {
    pub fn new(root: &Path, prefix: &str) -> Self {
        LocalConfig {
            root: root
                .to_str()
                .expect("LocalConfig root path invalid")
                .to_string(),
            prefix: prefix.to_string(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MemoryConfig;

impl MemoryConfig {
    pub fn new() -> Self {
        MemoryConfig {}
    }
}

#[derive(Clone, Debug)]
pub struct StorageConfig {
    pub local: LocalConfig,
    pub s3: S3Config,
    pub memory: MemoryConfig,
}

impl StorageConfig {
    pub fn new(local: LocalConfig, s3: S3Config) -> Self {
        StorageConfig {
            local,
            s3,
            memory: MemoryConfig::new(),
        }
    }
}

/// Enumeration of possible storage locations for resources.
///
/// This enum defines where ADMM problem data can be stored and accessed from.
/// The storage backend system automatically handles synchronization between
/// these different locations as needed.
///
/// # Storage Characteristics
///
/// - **Local**: Fast access, persistent across restarts, limited to single machine
/// - **S3**: Slower access, persistent, accessible from multiple machines
/// - **Memory**: Fastest access, ephemeral, limited by available RAM
///
/// # Example
///
/// ```rust
/// # use srsadmm_core::resource::ResourceLocation;
///
/// let fast_access = ResourceLocation::Memory;
/// let persistent = ResourceLocation::Local;
/// let distributed = ResourceLocation::S3;
/// ```
#[derive(Clone, Copy, Eq, PartialEq, Hash)]
pub enum ResourceLocation {
    /// Local filesystem storage with compression
    Local,
    /// Amazon S3 cloud storage
    S3,
    /// In-memory storage for fast access
    Memory,
    // TODO: implement non-copy locking memory strategy. May be useful for very large matrices,
    // but why not just do file operations?
    // MemoryLock
}

#[derive(Clone)]
pub(crate) struct ProblemResourceImpl<T: Clone + Serialize + DeserializeOwned + Send + Sync + 'static> {
    default_location: ResourceLocation,
    id: String,
    storage_config: StorageConfig,
    s3_backend: Arc<S3StorageBackend<T>>,
    local_backend: Arc<LocalStorageBackend<T>>,
    memory_backend: Arc<MemoryStorageBackend<T>>,
    updated: HashSet<ResourceLocation>,
    _marker: PhantomData<T>,
}

impl<T> ProblemResourceImpl<T>
where
    T: Serialize + DeserializeOwned + Clone + Send + Sync,
{
    pub fn new(
        id: String,
        default_location: ResourceLocation,
        storage_config: &StorageConfig,
    ) -> Self {
        let config = storage_config.clone();
        ProblemResourceImpl {
            default_location,
            id,
            s3_backend: Arc::new(S3StorageBackend::new(&config.s3)),
            local_backend: Arc::new(LocalStorageBackend::new(&config.local)),
            memory_backend: Arc::new(MemoryStorageBackend::new(&config.memory)),
            updated: HashSet::new(),
            _marker: PhantomData,
            storage_config: config,
        }
    }

    pub async fn new_from(
        id: String,
        default_location: ResourceLocation,
        storage_config: &StorageConfig,
        value: T,
    ) -> Result<Self, Box<dyn std::error::Error>>
    where
        T: Serialize + DeserializeOwned + Clone + Send + Sync,
    {
        let mut resource = Self::new(id, default_location, storage_config);
        let _ = resource.write(value).await.unwrap();
        Ok(resource)
    }

    pub async fn write(&mut self, value: T) -> Result<(), Box<dyn std::error::Error>> {
        self.updated.clear();
        match self.default_location {
            ResourceLocation::Local => {
                self.local_backend.write(&self.id, value.clone()).await?;
            }
            ResourceLocation::Memory => {
                self.memory_backend.write(&self.id, value.clone()).await?;
            }
            ResourceLocation::S3 => {
                // Write to temp file first in case the file is > 5Gb
                self.local_backend.write(&self.id, value.clone()).await?;
                self.updated.insert(ResourceLocation::Local);
                self.s3_backend
                    .copy_from(&self.id, self.local_backend.as_ref())
                    .await?;
            }
        }
        self.updated.insert(self.default_location);
        Ok(())
    }

    pub async fn read(&self) -> Result<T, Box<dyn std::error::Error>> {
        if self.updated.contains(&ResourceLocation::Memory) {
            // println!("[ProblemResourceImpl] Reading {} from memory", self.id);
            self.memory_backend.read(&self.id).await
        } else if self.updated.contains(&ResourceLocation::Local) {
            // println!("[ProblemResourceImpl] Reading {} from local", self.id);
            self.local_backend.read(&self.id).await
        } else if self.updated.contains(&ResourceLocation::S3) {
            // println!("[ProblemResourceImpl] Reading {} from s3", self.id);
            self.s3_backend.read(&self.id).await
        } else {
            return Err(format!("Resource {} not found in any storage location", self.id).into());
        }
    }

    pub async fn sync(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        self.sync_to(self.default_location).await
    }

    pub async fn sync_to(
        &mut self,
        location: ResourceLocation,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // If the location already has the most recent update, do nothing
        if self.updated.contains(&location) {
            return Ok(());
        }

        // Find the source location that has the most recent update
        let source_location = if self.updated.contains(&ResourceLocation::Local) {
            ResourceLocation::Local
        } else if self.updated.contains(&ResourceLocation::S3) {
            ResourceLocation::S3
        } else if self.updated.contains(&ResourceLocation::Memory) {
            ResourceLocation::Memory
        } else {
            return Err(format!("Resource {} not found in any storage location", self.id).into());
        };

        // Get the source and destination backends as trait objects
        let source_backend: &dyn StorageBackend<T> = match source_location {
            ResourceLocation::Local => self.local_backend.as_ref(),
            ResourceLocation::S3 => self.s3_backend.as_ref(),
            ResourceLocation::Memory => self.memory_backend.as_ref(),
        };

        let dest_backend: &dyn StorageBackend<T> = match location {
            ResourceLocation::Local => self.local_backend.as_ref(),
            ResourceLocation::S3 => self.s3_backend.as_ref(),
            ResourceLocation::Memory => self.memory_backend.as_ref(),
        };

        // Copy from source to destination
        dest_backend.copy_from(&self.id, source_backend).await?;

        // If we're syncing to S3 from memory, we should clean up the memory copy
        if location == ResourceLocation::S3 && source_location == ResourceLocation::Memory {
            self.memory_backend.delete(&self.id).await?;
        }

        // Mark the new location as updated
        self.updated.insert(location);
        Ok(())
    }

    // Read function that also caches the result to the specified location
    pub async fn read_and_cache(
        &mut self,
        location: ResourceLocation,
    ) -> Result<T, Box<dyn std::error::Error>> {
        match location {
            ResourceLocation::Local => {
                self.local_backend
                    .copy_from(&self.id, self.s3_backend.as_ref())
                    .await?;
            }
            ResourceLocation::Memory => {
                self.memory_backend
                    .copy_from(&self.id, self.s3_backend.as_ref())
                    .await?;
            }
            ResourceLocation::S3 => {
                self.s3_backend
                    .copy_from(&self.id, self.local_backend.as_ref())
                    .await?;
                // We will have a local copy available for future reads
                self.updated.insert(ResourceLocation::Local);
            }
        }
        self.updated.insert(location);
        Ok(self.read().await?)
    }

    pub fn s3_key(&self) -> String {
        self.s3_backend.key(&self.id)
    }

    pub fn s3_bucket(&self) -> String {
        self.s3_backend.bucket()
    }

    pub fn default_location(&self) -> ResourceLocation {
        self.default_location
    }

    pub fn storage_config(&self) -> &StorageConfig {
        &self.storage_config
    }

    pub fn set_default_location(&mut self, location: ResourceLocation) {
        self.default_location = location;
    }

    /// Mark a location as updated, useful when external processes (like Lambda) write to S3
    pub fn mark_updated(&mut self, location: ResourceLocation) {
        self.updated.clear();
        self.updated.insert(location);
    }

    pub fn local_path(&self) -> PathBuf {
        self.local_backend.file_path(&self.id)
    }
}
