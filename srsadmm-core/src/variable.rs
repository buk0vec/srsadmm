extern crate nalgebra as na;

use crate::{
    resource::{ProblemResourceImpl, ResourceLocation, StorageConfig},
    utils::LassoError,
};
use memmap2::Mmap;
use rayon::prelude::*;
use std::{fs::File, io::BufWriter, io::Write, path::Path};

/// A distributed matrix variable for ADMM optimization.
///
/// `MatrixVariable` represents a matrix that can be stored and synchronized
/// across different storage backends (local disk, S3, memory). It provides
/// a high-level interface for matrix operations commonly needed in ADMM
/// algorithms while handling the complexity of distributed storage.
///
/// # Storage Locations
///
/// Variables can be stored in multiple locations:
/// - **Local**: On local disk with compression
/// - **S3**: In AWS S3 for distributed access
/// - **Memory**: In RAM for fast access
///
/// # Example
///
/// ```rust,no_run
/// # use srsadmm_core::variable::MatrixVariable;
/// # use srsadmm_core::resource::{ResourceLocation, StorageConfig};
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// # let storage_config = StorageConfig::new(/* ... */);
///
/// // Create a 100x50 matrix of zeros
/// let mut matrix = MatrixVariable::zeros(
///     "my_matrix".to_string(),
///     100, 50,
///     ResourceLocation::Local,
///     &storage_config
/// ).await;
///
/// // Sync to S3 for distributed access
/// matrix.sync_to(ResourceLocation::S3).await?;
///
/// // Read the matrix data
/// let data = matrix.read().await?;
/// println!("Matrix shape: {}x{}", data.nrows(), data.ncols());
/// # Ok(())
/// # }
/// ```
#[derive(Clone)]
pub struct MatrixVariable {
    /// Unique identifier for this variable
    id: String,
    /// Underlying resource manager handling storage and synchronization
    _resource: ProblemResourceImpl<na::DMatrix<f32>>,
}

impl MatrixVariable {
    /// Creates a new matrix variable with the specified configuration.
    ///
    /// # Arguments
    ///
    /// * `id` - Unique identifier for this variable
    /// * `default_location` - Primary storage location for the variable
    /// * `storage_config` - Configuration for all storage backends
    ///
    /// # Returns
    ///
    /// A new `MatrixVariable` instance ready for use
    pub fn new(
        id: String,
        default_location: ResourceLocation,
        storage_config: &StorageConfig,
    ) -> Self {
        MatrixVariable {
            id: id.clone(),
            _resource: ProblemResourceImpl::new(id, default_location, storage_config),
        }
    }

    /// Returns the unique identifier for this variable.
    ///
    /// # Returns
    ///
    /// A string slice containing the variable's ID
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Creates a matrix variable from an existing matrix.
    ///
    /// This constructor immediately writes the provided matrix to storage
    /// at the specified default location.
    ///
    /// # Arguments
    ///
    /// * `id` - Unique identifier for this variable
    /// * `default_location` - Primary storage location for the variable
    /// * `storage_config` - Configuration for all storage backends
    /// * `matrix` - The initial matrix data to store
    ///
    /// # Returns
    ///
    /// A new `MatrixVariable` containing the provided matrix data
    pub async fn from_matrix(
        id: String,
        default_location: ResourceLocation,
        storage_config: &StorageConfig,
        matrix: na::DMatrix<f32>,
    ) -> Self {
        let mut variable = MatrixVariable::new(id, default_location, storage_config);
        let _ = variable._resource.write(matrix).await.unwrap();
        variable
    }

    /// Creates a matrix variable initialized with zeros.
    ///
    /// # Arguments
    ///
    /// * `id` - Unique identifier for this variable
    /// * `nrows` - Number of rows in the matrix
    /// * `ncols` - Number of columns in the matrix
    /// * `default_location` - Primary storage location for the variable
    /// * `storage_config` - Configuration for all storage backends
    ///
    /// # Returns
    ///
    /// A new `MatrixVariable` containing a zero matrix of the specified dimensions
    pub async fn zeros(
        id: String,
        nrows: usize,
        ncols: usize,
        default_location: ResourceLocation,
        storage_config: &StorageConfig,
    ) -> Self {
        let matrix = na::DMatrix::<f32>::zeros(nrows, ncols);
        MatrixVariable::from_matrix(id, default_location, storage_config, matrix).await
    }

    /// Creates a matrix variable initialized with ones.
    ///
    /// # Arguments
    ///
    /// * `id` - Unique identifier for this variable
    /// * `nrows` - Number of rows in the matrix
    /// * `ncols` - Number of columns in the matrix
    /// * `default_location` - Primary storage location for the variable
    /// * `storage_config` - Configuration for all storage backends
    ///
    /// # Returns
    ///
    /// A new `MatrixVariable` containing a matrix of ones with the specified dimensions
    pub async fn ones(
        id: String,
        nrows: usize,
        ncols: usize,
        default_location: ResourceLocation,
        storage_config: &StorageConfig,
    ) -> Self {
        let matrix = na::DMatrix::<f32>::from_element(nrows, ncols, 1.0);
        MatrixVariable::from_matrix(id, default_location, storage_config, matrix).await
    }

    /// Creates a matrix variable initialized as an identity matrix.
    ///
    /// # Arguments
    ///
    /// * `id` - Unique identifier for this variable
    /// * `nrows` - Number of rows in the matrix
    /// * `ncols` - Number of columns in the matrix
    /// * `default_location` - Primary storage location for the variable
    /// * `storage_config` - Configuration for all storage backends
    ///
    /// # Returns
    ///
    /// A new `MatrixVariable` containing an identity matrix of the specified dimensions
    pub async fn eye(
        id: String,
        nrows: usize,
        ncols: usize,
        default_location: ResourceLocation,
        storage_config: &StorageConfig,
    ) -> Self {
        let matrix = na::DMatrix::identity(nrows, ncols);
        MatrixVariable::from_matrix(id, default_location, storage_config, matrix).await
    }

    /// Synchronizes the matrix variable to the default location.
    ///
    /// # Returns
    ///
    /// * `Ok(())` if the synchronization is successful.
    /// * `Err(Box<dyn std::error::Error>)` if the synchronization fails.
    ///
    pub async fn sync(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        self._resource.sync().await
    }

    /// Synchronizes the matrix variable to a specific location.
    ///
    /// # Arguments
    ///
    /// * `location` - The storage location to synchronize to.
    ///
    /// # Returns
    ///
    /// * `Ok(())` if the synchronization is successful.
    /// * `Err(Box<dyn std::error::Error>)` if the synchronization fails.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use srsadmm_core::variable::MatrixVariable;
    /// use srsadmm_core::resource::ResourceLocation;
    /// async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut matrix = MatrixVariable::new("my_matrix".to_string(), ResourceLocation::Local, &storage_config);
    /// matrix.sync_to(ResourceLocation::S3).await?;
    /// Ok(())
    /// }
    /// ```
    pub async fn sync_to(
        &mut self,
        location: ResourceLocation,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self._resource.sync_to(location).await
    }

    /// Returns the default storage location for this variable.
    ///
    /// # Returns
    ///
    /// The default storage location for this variable
    pub fn default_location(&self) -> ResourceLocation {
        self._resource.default_location()
    }

    /// Reads the matrix data from storage.
    ///
    /// This method reads the matrix from the most recently updated storage location,
    /// preferring memory > local > S3 for performance.
    ///
    /// # Returns
    ///
    /// * `Ok(DMatrix<f32>)` - The matrix data
    /// * `Err(Box<dyn Error>)` - If reading fails or no valid data is found
    pub async fn read(&self) -> Result<na::DMatrix<f32>, Box<dyn std::error::Error>> {
        self._resource.read().await
    }

    /// Reads and caches the matrix data from a specific location.
    ///
    /// This method first synchronizes the variable to the specified location,
    /// then reads and returns the matrix data.
    ///
    /// # Arguments
    ///
    /// * `location` - The storage location to read from
    ///
    /// # Returns
    ///
    /// * `Ok(DMatrix<f32>)` - The matrix data
    /// * `Err(Box<dyn Error>)` - If synchronization or reading fails
    pub async fn read_and_cache(
        &mut self,
        location: ResourceLocation,
    ) -> Result<na::DMatrix<f32>, Box<dyn std::error::Error>> {
        self._resource.read_and_cache(location).await
    }

    /// Writes new matrix data to the default storage location.
    ///
    /// This method overwrites the current matrix data with the provided matrix
    /// and marks the default location as having the most recent update.
    ///
    /// # Arguments
    ///
    /// * `matrix` - The new matrix data to store
    ///
    /// # Returns
    ///
    /// * `Ok(())` - If writing succeeds
    /// * `Err(Box<dyn Error>)` - If writing fails
    pub async fn write(
        &mut self,
        matrix: na::DMatrix<f32>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self._resource.write(matrix).await
    }

    pub fn s3_key(&self) -> String {
        self._resource.s3_key()
    }

    pub fn s3_bucket(&self) -> String {
        self._resource.s3_bucket()
    }

    pub fn set_default_location(&mut self, location: ResourceLocation) {
        self._resource.set_default_location(location);
    }

    /// Mark this variable as updated in a specific location
    /// Useful when external processes (like Lambda) write to S3
    pub fn mark_updated(&mut self, location: ResourceLocation) {
        self._resource.mark_updated(location);
    }

    pub async fn sub(
        &mut self,
        other: &mut MatrixVariable,
    ) -> Result<MatrixVariable, Box<dyn std::error::Error>> {
        // Ensure both variables are in the same location (use self's default location)
        let target_location = self.default_location();
        self._resource.sync_to(target_location).await?;
        other._resource.sync_to(target_location).await?;

        let self_matrix = self._resource.read().await?;
        let other_matrix = other._resource.read().await?;
        let result = self_matrix - other_matrix;
        Ok(MatrixVariable::from_matrix(
            format!("{}-sub-{}", self.id, other.id),
            target_location,
            self._resource.storage_config(),
            result,
        )
        .await)
    }

    pub async fn add(
        &mut self,
        other: &mut MatrixVariable,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Ensure both variables are in the same location (use self's default location)
        let target_location = self.default_location();
        self._resource.sync_to(target_location).await?;
        other._resource.sync_to(target_location).await?;

        let self_matrix = self._resource.read().await?;
        let other_matrix = other._resource.read().await?;
        let result = self_matrix + other_matrix;
        self._resource.write(result).await?;
        Ok(())
    }

    pub async fn add_identity(&mut self, factor: f32) -> Result<(), Box<dyn std::error::Error>> {
        self._resource.sync().await?;
        let matrix = self._resource.read().await?;
        let nrows = matrix.nrows();
        let ncols = matrix.ncols();
        let eye = na::DMatrix::<f32>::identity(nrows, ncols);
        let result = matrix + eye * factor;
        self._resource.write(result).await?;
        Ok(())
    }

    pub async fn inverse(&mut self) -> Result<MatrixVariable, Box<dyn std::error::Error>> {
        self._resource.sync().await?;
        let matrix = self._resource.read().await?;
        let inverse = matrix.try_inverse().ok_or("Matrix is not invertible")?;
        Ok(MatrixVariable::from_matrix(
            format!("{}-inverse", self.id),
            self.default_location(),
            self._resource.storage_config(),
            inverse,
        )
        .await)
    }

    pub async fn map_rows_by_subproblem<R, F>(
        &self,
        n_subproblems: usize,
        mut f: F,
    ) -> Result<Vec<R>, Box<dyn std::error::Error>>
    where
        F: FnMut(na::DMatrix<f32>, usize) -> R,
    {
        let matrix = self._resource.read().await?;
        let nrows = matrix.nrows();
        let mut results = Vec::new();

        let base_chunk_size = nrows / n_subproblems;
        let remainder = nrows % n_subproblems;

        let mut current_row = 0;

        for i in 0..n_subproblems {
            let chunk_size = if i < remainder {
                base_chunk_size + 1
            } else {
                base_chunk_size
            };
            let chunk = matrix.rows(current_row, chunk_size).clone_owned();
            results.push(f(chunk, i));
            current_row += chunk_size;
        }

        Ok(results)
    }

    pub async fn map_rows_by_subproblem_with<R, F>(
        &self,
        var2: &MatrixVariable,
        n_subproblems: usize,
        mut f: F,
    ) -> Result<Vec<R>, Box<dyn std::error::Error>>
    where
        F: FnMut(na::DMatrix<f32>, na::DMatrix<f32>, usize) -> R,
    {
        let matrix = self._resource.read().await?;
        let matrix2 = var2._resource.read().await?;
        let nrows = matrix.nrows();
        let nrows2 = matrix2.nrows();
        let ncols = matrix.ncols();
        let ncols2 = matrix2.ncols();
        let mut results = Vec::new();

        if nrows != nrows2 {
            return Err(LassoError::from_string(format!(
                "Matrix rows do not match: {}x{} and {}x{}",
                nrows, ncols, nrows2, ncols2
            ))
            .into());
        }

        let base_chunk_size = nrows / n_subproblems;
        let remainder = nrows % n_subproblems;

        let mut current_row = 0;

        for i in 0..n_subproblems {
            let chunk_size = if i < remainder {
                base_chunk_size + 1
            } else {
                base_chunk_size
            };
            let chunk = matrix.rows(current_row, chunk_size).clone_owned();
            let chunk2 = matrix2.rows(current_row, chunk_size).clone_owned();
            results.push(f(chunk, chunk2, i));
            current_row += chunk_size;
        }
        Ok(results)
    }

    pub async fn map_rows_by_subproblem_with_rayon<R, F>(
        &self,
        var2: &MatrixVariable,
        n_subproblems: usize,
        n_threads: usize,
        f: F,
    ) -> Result<Vec<R>, Box<dyn std::error::Error>>
    where
        F: Fn(na::DMatrix<f32>, na::DMatrix<f32>, usize) -> R + Send + Sync,
        R: Send,
    {
        let matrix = self._resource.read().await?;
        let matrix2 = var2._resource.read().await?;
        let nrows = matrix.nrows();
        let nrows2 = matrix2.nrows();
        let ncols = matrix.ncols();
        let ncols2 = matrix2.ncols();

        if nrows != nrows2 {
            return Err(LassoError::from_string(format!(
                "Matrix rows do not match: {}x{} and {}x{}",
                nrows, ncols, nrows2, ncols2
            ))
            .into());
        }

        let base_chunk_size = nrows / n_subproblems;
        let remainder = nrows % n_subproblems;

        // Precompute chunk info
        let chunk_data = (0..n_subproblems)
            .map(|i| {
                let chunk_size = if i < remainder {
                    base_chunk_size + 1
                } else {
                    base_chunk_size
                };
                let chunk_index = base_chunk_size * i + if i < remainder { i } else { remainder };
                (chunk_index, chunk_size)
            })
            .collect::<Vec<_>>();

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(n_threads)
            .build()
            .unwrap();

        let results = pool.install(|| {
            chunk_data
                .into_par_iter()
                .zip(0..n_subproblems)
                .map(|((chunk_index, chunk_size), i)| {
                    let chunk = matrix.rows(chunk_index, chunk_size).clone_owned();
                    let chunk2 = matrix2.rows(chunk_index, chunk_size).clone_owned();
                    f(chunk, chunk2, i)
                })
                .collect::<Vec<_>>()
        });

        Ok(results)
    }

    pub async fn map_rows_by_subproblem_with_async<R, F>(
        &self,
        var2: &MatrixVariable,
        n_subproblems: usize,
        f: F,
    ) -> Result<Vec<R>, Box<dyn std::error::Error>>
    where
        F: AsyncFn(na::DMatrix<f32>, na::DMatrix<f32>, usize) -> R,
    {
        let matrix = self._resource.read().await?;
        let matrix2 = var2._resource.read().await?;
        let nrows = matrix.nrows();
        let nrows2 = matrix2.nrows();
        let ncols = matrix.ncols();
        let ncols2 = matrix2.ncols();

        if nrows != nrows2 {
            return Err(LassoError::from_string(format!(
                "Matrix rows do not match: {}x{} and {}x{}",
                nrows, ncols, nrows2, ncols2
            ))
            .into());
        }

        let base_chunk_size = nrows / n_subproblems;
        let remainder = nrows % n_subproblems;

        let mut current_row = 0;

        let futures = (0..n_subproblems)
            .map(|i| {
                let chunk_size = if i < remainder {
                    base_chunk_size + 1
                } else {
                    base_chunk_size
                };
                let chunk = matrix.rows(current_row, chunk_size).clone_owned();
                let chunk2 = matrix2.rows(current_row, chunk_size).clone_owned();
                current_row += chunk_size;
                f(chunk, chunk2, i)
            })
            .collect::<Vec<_>>();

        let results = futures::future::join_all(futures).await;
        Ok(results)
    }
}

/// A distributed scalar variable for ADMM optimization.
///
/// `ScalarVariable` represents a single floating-point value that can be stored
/// and synchronized across different storage backends (local disk, S3, memory).
/// This is useful for storing parameters, objective values, or other scalar
/// quantities in distributed ADMM algorithms.
///
/// # Example
///
/// ```rust,no_run
/// # use srsadmm_core::variable::ScalarVariable;
/// # use srsadmm_core::resource::{ResourceLocation, StorageConfig};
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// # let storage_config = StorageConfig::new(/* ... */);
///
/// // Create a scalar variable with initial value
/// let mut scalar = ScalarVariable::from_scalar(
///     "lambda".to_string(),
///     ResourceLocation::Local,
///     &storage_config,
///     0.1
/// ).await;
///
/// // Read the current value
/// let value = scalar.read().await?;
/// println!("Current value: {}", value);
///
/// // Update the value
/// scalar.write(0.2).await?;
/// # Ok(())
/// # }
/// ```
pub struct ScalarVariable {
    /// Unique identifier for this scalar variable
    #[allow(dead_code)]
    id: String,
    /// Underlying resource manager handling storage and synchronization
    _resource: ProblemResourceImpl<f32>,
}

impl ScalarVariable {
    /// Creates a new scalar variable with the specified configuration.
    ///
    /// # Arguments
    ///
    /// * `id` - Unique identifier for this variable
    /// * `default_location` - Primary storage location for the variable
    /// * `storage_config` - Configuration for all storage backends
    ///
    /// # Returns
    ///
    /// A new `ScalarVariable` instance ready for use
    pub fn new(
        id: String,
        default_location: ResourceLocation,
        storage_config: &StorageConfig,
    ) -> Self {
        ScalarVariable {
            id: id.clone(),
            _resource: ProblemResourceImpl::new(id, default_location, storage_config),
        }
    }

    /// Creates a scalar variable from an initial value.
    ///
    /// This constructor immediately writes the provided scalar value to storage
    /// at the specified default location.
    ///
    /// # Arguments
    ///
    /// * `id` - Unique identifier for this variable
    /// * `default_location` - Primary storage location for the variable
    /// * `storage_config` - Configuration for all storage backends
    /// * `scalar` - The initial scalar value to store
    ///
    /// # Returns
    ///
    /// A new `ScalarVariable` containing the provided scalar value
    pub async fn from_scalar(
        id: String,
        default_location: ResourceLocation,
        storage_config: &StorageConfig,
        scalar: f32,
    ) -> Self {
        let mut variable = ScalarVariable::new(id, default_location, storage_config);
        let _ = variable._resource.write(scalar).await.unwrap();
        variable
    }

    pub async fn read(&self) -> Result<f32, Box<dyn std::error::Error>> {
        self._resource.read().await
    }

    pub async fn read_and_cache(
        &mut self,
        location: ResourceLocation,
    ) -> Result<f32, Box<dyn std::error::Error>> {
        self._resource.read_and_cache(location).await
    }

    pub async fn write(&mut self, scalar: f32) -> Result<(), Box<dyn std::error::Error>> {
        self._resource.write(scalar).await
    }

    pub fn s3_key(&self) -> String {
        self._resource.s3_key()
    }

    pub fn s3_bucket(&self) -> String {
        self._resource.s3_bucket()
    }
}

#[derive(Clone, Copy, PartialEq)]
pub enum MatrixStorageType {
    Columns = 0,
    Rows = 1,
}

impl std::fmt::Display for MatrixStorageType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MatrixStorageType::Columns => write!(f, "Columns"),
            MatrixStorageType::Rows => write!(f, "Rows"),
        }
    }
}

/// DataMatrixVariables are variables that store very large matrices that require
/// quick row/column slice access. They are only stored on the local file system
/// and don't have a direct "read()" method, but instead have methods for reading
/// and iterating over chunks of rows/columns.
pub struct DataMatrixVariable {
    id: String,
    _resource: ProblemResourceImpl<na::DMatrix<f32>>,
    storage_type: MatrixStorageType,
    nrows: usize,
    ncols: usize,
}

impl DataMatrixVariable {
    pub(crate) fn new(
        id: String,
        storage_config: &StorageConfig,
        storage_type: MatrixStorageType,
        nrows: usize,
        ncols: usize,
    ) -> Self {
        DataMatrixVariable {
            id: id.clone(),
            _resource: ProblemResourceImpl::new(id, ResourceLocation::Local, storage_config),
            storage_type,
            nrows,
            ncols,
        }
    }

    pub fn from_matrix(
        id: String,
        matrix: na::DMatrix<f32>,
        storage_type: MatrixStorageType,
        storage_config: &StorageConfig,
    ) -> Self {
        let var = DataMatrixVariable::new(
            id,
            storage_config,
            storage_type,
            matrix.nrows(),
            matrix.ncols(),
        );
        let mut path = var._resource.local_path();
        // path hack, this isn't compressed
        path.set_extension("");
        write_data_matrix_to_file(&matrix, &path, storage_type).unwrap();
        var
    }

    pub fn from_problem_file(id: String, file_path: &Path, storage_config: &StorageConfig) -> Self {
        let new_path = Path::new(&storage_config.local.root)
            .join(format!("{}{}", storage_config.local.prefix, id))
            .with_extension("bin");
        std::fs::copy(file_path, &new_path).unwrap();
        let file = File::open(new_path).unwrap();
        let mmap = unsafe { Mmap::map(&file).unwrap() };
        let header = mmap[0..24].to_vec();
        let nrows = u64::from_le_bytes(header[8..16].try_into().unwrap()) as usize;
        let ncols = u64::from_le_bytes(header[16..24].try_into().unwrap()) as usize;
        let storage_type = match header[0] {
            0 => MatrixStorageType::Columns,
            1 => MatrixStorageType::Rows,
            _ => panic!("Invalid storage type"),
        };
        let var = DataMatrixVariable::new(id, storage_config, storage_type, nrows, ncols);
        var
    }

    pub fn read_chunk(
        &self,
        chunk_index: usize,
        chunk_size: usize,
    ) -> Result<na::DMatrix<f32>, Box<dyn std::error::Error>> {
        let mut path = self._resource.local_path();
        path.set_extension(""); // Not using zstd for datamatrices but backend adds it to extension
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        let header = mmap[0..24].to_vec();

        let nrows = u64::from_le_bytes(header[8..16].try_into().unwrap()) as usize;
        let ncols = u64::from_le_bytes(header[16..24].try_into().unwrap()) as usize;

        return match self.storage_type {
            MatrixStorageType::Columns => {
                if chunk_index + chunk_size > ncols {
                    return Err(LassoError::from_string(format!(
                        "Chunk index and size exceed number of columns: {} + {} > {}",
                        chunk_index, chunk_size, ncols
                    ))
                    .into());
                }

                let start_index = 24 + chunk_index * 4 * nrows;
                let end_index = start_index + chunk_size * 4 * nrows;
                let data = &mmap[start_index..end_index];
                let data_f32: &[f32] = unsafe {
                    std::slice::from_raw_parts(data.as_ptr() as *const f32, nrows * chunk_size)
                };

                let matrix = na::DMatrix::<f32>::from_column_slice(nrows, chunk_size, data_f32);
                Ok(matrix)
            }
            MatrixStorageType::Rows => {
                if chunk_index + chunk_size > nrows {
                    return Err(LassoError::from_string(format!(
                        "Chunk index and size exceed number of rows: {} + {} > {}",
                        chunk_index, chunk_size, nrows
                    ))
                    .into());
                }
                let start_index = 24 + chunk_index * ncols * 4;
                let end_index = start_index + chunk_size * ncols * 4;
                let data = &mmap[start_index..end_index];
                let data_f32: &[f32] = unsafe {
                    std::slice::from_raw_parts(data.as_ptr() as *const f32, chunk_size * ncols)
                };
                let matrix = na::DMatrix::<f32>::from_row_slice(chunk_size, ncols, data_f32);
                Ok(matrix)
            }
        };
    }

    pub fn iter_subproblems<T, F>(
        &self,
        n_subproblems: usize,
        f: F,
    ) -> Result<Vec<T>, Box<dyn std::error::Error>>
    where
        F: Fn(na::DMatrix<f32>, usize) -> T,
    {
        let base_chunk_size = self.nrows() / n_subproblems;
        let remainder = self.nrows() % n_subproblems;

        let chunk_data = (0..n_subproblems)
            .map(|i| {
                let chunk_size = if i < remainder {
                    base_chunk_size + 1
                } else {
                    base_chunk_size
                };
                let chunk_index = base_chunk_size * i + if i < remainder { i } else { remainder };
                (chunk_index, chunk_size)
            })
            .collect::<Vec<_>>();

        let results = chunk_data
            .into_iter()
            .enumerate()
            .map(|(i, (chunk_index, chunk_size))| {
                let chunk = self
                    .read_chunk(chunk_index, chunk_size)
                    .expect("Failed to read chunk");
                f(chunk, i)
            })
            .collect::<Vec<_>>();

        Ok(results)
    }

    pub fn iter_subproblems_with<T, F>(
        &self,
        var2: &DataMatrixVariable,
        n_subproblems: usize,
        f: F,
    ) -> Result<Vec<T>, Box<dyn std::error::Error>>
    where
        F: Fn(na::DMatrix<f32>, na::DMatrix<f32>, usize) -> T,
    {
        let base_chunk_size = self.nrows() / n_subproblems;
        let remainder = self.nrows() % n_subproblems;

        if self.storage_type != var2.storage_type {
            return Err(LassoError::from_string(format!(
                "Storage types do not match: {} and {}",
                self.storage_type, var2.storage_type
            ))
            .into());
        }

        match self.storage_type {
            MatrixStorageType::Columns => {
                if self.ncols() != var2.ncols() {
                    return Err(LassoError::from_string(format!(
                        "Number of columns do not match: {} and {}",
                        self.ncols(),
                        var2.ncols()
                    ))
                    .into());
                }
            }
            MatrixStorageType::Rows => {
                if self.nrows() != var2.nrows() {
                    return Err(LassoError::from_string(format!(
                        "Number of rows do not match: {} and {}",
                        self.nrows(),
                        var2.nrows()
                    ))
                    .into());
                }
            }
        }

        let chunk_data = (0..n_subproblems)
            .map(|i| {
                let chunk_size = if i < remainder {
                    base_chunk_size + 1
                } else {
                    base_chunk_size
                };
                let chunk_index = base_chunk_size * i + if i < remainder { i } else { remainder };
                (chunk_index, chunk_size)
            })
            .collect::<Vec<_>>();

        let results = chunk_data
            .into_iter()
            .enumerate()
            .map(|(i, (chunk_index, chunk_size))| {
                let chunk = self
                    .read_chunk(chunk_index, chunk_size)
                    .expect("Failed to read chunk");
                let chunk2 = var2
                    .read_chunk(chunk_index, chunk_size)
                    .expect("Failed to read chunk");
                f(chunk, chunk2, i)
            })
            .collect::<Vec<_>>();

        Ok(results)
    }

    pub fn iter_subproblems_rayon<T, F>(
        &self,
        n_subproblems: usize,
        n_threads: usize,
        f: F,
    ) -> Result<Vec<T>, Box<dyn std::error::Error>>
    where
        F: Fn(na::DMatrix<f32>, usize) -> T + Send + Sync,
        T: Send,
    {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(n_threads)
            .build()
            .unwrap();
        let base_chunk_size = self.nrows() / n_subproblems;
        let remainder = self.nrows() % n_subproblems;

        let chunk_data = (0..n_subproblems)
            .map(|i| {
                let chunk_size = if i < remainder {
                    base_chunk_size + 1
                } else {
                    base_chunk_size
                };
                let chunk_index = base_chunk_size * i + if i < remainder { i } else { remainder };
                (chunk_index, chunk_size)
            })
            .collect::<Vec<_>>();

        let results = pool.install(|| {
            chunk_data
                .into_par_iter()
                .zip(0..n_subproblems)
                .map(|((chunk_index, chunk_size), i)| {
                    let chunk = self
                        .read_chunk(chunk_index, chunk_size)
                        .expect("Failed to read chunk");
                    f(chunk, i)
                })
                .collect::<Vec<_>>()
        });
        Ok(results)
    }

    pub fn iter_subproblems_with_rayon<T, F>(
        &self,
        var2: &DataMatrixVariable,
        n_subproblems: usize,
        n_threads: usize,
        f: F,
    ) -> Result<Vec<T>, Box<dyn std::error::Error>>
    where
        F: Fn(na::DMatrix<f32>, na::DMatrix<f32>, usize) -> T + Send + Sync,
        T: Send,
    {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(n_threads)
            .build()
            .unwrap();
        let base_chunk_size = self.nrows() / n_subproblems;
        let remainder = self.nrows() % n_subproblems;

        if self.storage_type != var2.storage_type {
            return Err(LassoError::from_string(format!(
                "Storage types do not match: {} and {}",
                self.storage_type, var2.storage_type
            ))
            .into());
        }

        match self.storage_type {
            MatrixStorageType::Columns => {
                if self.ncols() != var2.ncols() {
                    return Err(LassoError::from_string(format!(
                        "Number of columns do not match: {} and {}",
                        self.ncols(),
                        var2.ncols()
                    ))
                    .into());
                }
            }
            MatrixStorageType::Rows => {
                if self.nrows() != var2.nrows() {
                    return Err(LassoError::from_string(format!(
                        "Number of rows do not match: {} and {}",
                        self.nrows(),
                        var2.nrows()
                    ))
                    .into());
                }
            }
        }

        let chunk_data = (0..n_subproblems)
            .map(|i| {
                let chunk_size = if i < remainder {
                    base_chunk_size + 1
                } else {
                    base_chunk_size
                };
                let chunk_index = base_chunk_size * i + if i < remainder { i } else { remainder };
                (chunk_index, chunk_size)
            })
            .collect::<Vec<_>>();

        let results = pool.install(|| {
            chunk_data
                .into_par_iter()
                .zip(0..n_subproblems)
                .map(|((chunk_index, chunk_size), i)| {
                    let chunk = self
                        .read_chunk(chunk_index, chunk_size)
                        .expect("Failed to read chunk");
                    let chunk2 = var2
                        .read_chunk(chunk_index, chunk_size)
                        .expect("Failed to read chunk");
                    f(chunk, chunk2, i)
                })
                .collect::<Vec<_>>()
        });
        Ok(results)
    }

    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn nrows(&self) -> usize {
        self.nrows
    }

    pub fn ncols(&self) -> usize {
        self.ncols
    }
}

// Custom matrix format:
// 1. Header: 8 bytes for storage type (0 = columns, 1 = rows)
// 2. Header: 8 bytes for number of rows
// 3. Header: 8 bytes for number of columns
// 4. Data: matrix data in row or column major order

pub(crate) fn write_data_matrix_to_file(
    matrix: &na::DMatrix<f32>,
    file_path: &Path,
    storage_type: MatrixStorageType,
) -> Result<(), Box<dyn std::error::Error>> {
    let batch_size = 1024;
    let file = File::create(file_path)?;
    let nrows = matrix.nrows() as u64;
    let ncols = matrix.ncols() as u64;
    let buf_size = match storage_type {
        MatrixStorageType::Columns => nrows * 4,
        MatrixStorageType::Rows => ncols * 4,
    };
    let mut writer = BufWriter::with_capacity(buf_size as usize, file);
    let header = u64_to_bytes(storage_type as u64);
    writer.write_all(&header)?;
    let header = [nrows, ncols];
    let header = header
        .iter()
        .map(|&v| u64_to_bytes(v))
        .flatten()
        .collect::<Vec<_>>();
    writer.write_all(&header)?;
    let batches = match storage_type {
        MatrixStorageType::Columns => (0..matrix.ncols()).step_by(batch_size),
        MatrixStorageType::Rows => (0..matrix.nrows()).step_by(batch_size),
    };
    match storage_type {
        MatrixStorageType::Columns => {
            for col in batches {
                let start_col = col;
                let n_cols = std::cmp::min(start_col + batch_size, matrix.ncols()) - start_col;
                let col_view = matrix.columns(start_col, n_cols).clone_owned();
                let col_data = col_view.as_slice();
                let byte_slice = unsafe {
                    std::slice::from_raw_parts(col_data.as_ptr() as *const u8, col_data.len() * 4)
                };
                writer.write_all(byte_slice)?;
            }
        }
        MatrixStorageType::Rows => {
            for row in batches {
                let start_row = row;
                let n_rows = std::cmp::min(start_row + batch_size, matrix.nrows()) - start_row;
                let row_data_t = matrix.rows(start_row, n_rows).clone_owned().transpose();
                let row_data = row_data_t.as_slice();
                let byte_slice = unsafe {
                    std::slice::from_raw_parts(row_data.as_ptr() as *const u8, row_data.len() * 4)
                };
                writer.write_all(byte_slice)?;
            }
        }
    }
    Ok(())
}

pub(crate) fn u64_to_bytes(value: u64) -> [u8; 8] {
    let mut bytes = [0; 8];
    bytes.copy_from_slice(&value.to_le_bytes());
    bytes
}
