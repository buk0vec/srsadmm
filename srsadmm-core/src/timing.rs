use crate::utils::LassoError;
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    fs::OpenOptions,
    io::Write,
    time::{Duration, Instant},
};

/// A record of timing information for an ADMM algorithm step.
///
/// This structure captures detailed timing information for individual
/// steps within ADMM iterations, useful for performance analysis and
/// optimization identification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimingRecord {
    /// Name of the ADMM step (e.g., "update_x", "update_z")
    pub step_name: String,
    /// The iteration number when this step was executed
    pub iteration: usize,
    /// Duration of the step in milliseconds
    pub duration_ms: f64,
    /// Unix timestamp when the step was recorded
    pub timestamp: u64,
}

/// A record of timing information for Lambda function calls.
///
/// This structure captures timing data for distributed Lambda function
/// invocations, enabling analysis of cloud compute performance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LambdaTimingRecord {
    /// Name of the Lambda function that was called
    pub function_name: String,
    /// Type of operation performed by the Lambda function
    pub operation_type: String,
    /// The iteration number when this Lambda was called
    pub iteration: usize,
    /// Duration of the Lambda call in milliseconds (including network overhead)
    pub duration_ms: f64,
    /// Unix timestamp when the call was recorded
    pub timestamp: u64,
}

/// Performance tracking system for ADMM algorithms.
///
/// `TimingTracker` collects detailed timing information for both local
/// algorithm steps and distributed Lambda function calls. It provides
/// statistical analysis and CSV export capabilities for performance
/// optimization and bottleneck identification.
///
/// This is used internally by the ADMMSolver and Lambda functions to track
/// timing information automatically.
pub struct TimingTracker {
    /// Collection of step timing records
    step_timings: Vec<TimingRecord>,
    /// Collection of Lambda timing records
    lambda_timings: Vec<LambdaTimingRecord>,
    /// Current iteration number for new recordings
    current_iteration: usize,
}

impl TimingTracker {
    pub fn new() -> Self {
        Self {
            step_timings: Vec::new(),
            lambda_timings: Vec::new(),
            current_iteration: 0,
        }
    }

    pub fn start_iteration(&mut self) {
        self.current_iteration += 1;
    }

    pub fn record_step(&mut self, step_name: &str, duration: Duration) {
        let record = TimingRecord {
            step_name: step_name.to_string(),
            iteration: self.current_iteration,
            duration_ms: duration.as_secs_f64() * 1000.0,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        };
        self.step_timings.push(record);
    }

    pub fn record_lambda(&mut self, function_name: &str, operation_type: &str, duration: Duration) {
        let record = LambdaTimingRecord {
            function_name: function_name.to_string(),
            operation_type: operation_type.to_string(),
            iteration: self.current_iteration,
            duration_ms: duration.as_secs_f64() * 1000.0,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        };
        self.lambda_timings.push(record);
    }

    pub fn write_step_timings_to_csv(&self, filename: &str) -> Result<(), LassoError> {
        let mut file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(filename)
            .map_err(|e| LassoError::from_string(format!("Failed to create timing file: {}", e)))?;

        // Write CSV header
        writeln!(file, "step_name,iteration,duration_ms,timestamp")
            .map_err(|e| LassoError::from_string(format!("Failed to write header: {}", e)))?;

        // Write timing records
        for record in &self.step_timings {
            writeln!(
                file,
                "{},{},{:.3},{}",
                record.step_name, record.iteration, record.duration_ms, record.timestamp
            )
            .map_err(|e| {
                LassoError::from_string(format!("Failed to write timing record: {}", e))
            })?;
        }

        Ok(())
    }

    pub fn write_lambda_timings_to_csv(&self, filename: &str) -> Result<(), LassoError> {
        let mut file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(filename)
            .map_err(|e| {
                LassoError::from_string(format!("Failed to create lambda timing file: {}", e))
            })?;

        // Write CSV header
        writeln!(
            file,
            "function_name,operation_type,iteration,duration_ms,timestamp"
        )
        .map_err(|e| LassoError::from_string(format!("Failed to write header: {}", e)))?;

        // Write timing records
        for record in &self.lambda_timings {
            writeln!(
                file,
                "{},{},{},{:.3},{}",
                record.function_name,
                record.operation_type,
                record.iteration,
                record.duration_ms,
                record.timestamp
            )
            .map_err(|e| {
                LassoError::from_string(format!("Failed to write lambda timing record: {}", e))
            })?;
        }

        Ok(())
    }

    pub fn get_step_statistics(&self) -> HashMap<String, (f64, f64, usize)> {
        let mut stats = HashMap::new();

        for record in &self.step_timings {
            let entry = stats
                .entry(record.step_name.clone())
                .or_insert((0.0f64, 0.0f64, 0));
            entry.0 += record.duration_ms;
            entry.1 = entry.1.max(record.duration_ms);
            entry.2 += 1;
        }

        // Convert to (average, max, count)
        for (_, stats) in stats.iter_mut() {
            stats.0 /= stats.2 as f64;
        }

        stats
    }

    pub fn get_lambda_statistics(&self) -> HashMap<String, (f64, f64, usize)> {
        let mut stats = HashMap::new();

        for record in &self.lambda_timings {
            let key = format!("{}:{}", record.function_name, record.operation_type);
            let entry = stats.entry(key).or_insert((0.0f64, 0.0f64, 0));
            entry.0 += record.duration_ms;
            entry.1 = entry.1.max(record.duration_ms);
            entry.2 += 1;
        }

        // Convert to (average, max, count)
        for (_, stats) in stats.iter_mut() {
            stats.0 /= stats.2 as f64;
        }

        stats
    }
}

pub async fn time_async_fn<F, R>(
    tracker: &mut TimingTracker,
    name: &str,
    f: F,
) -> Result<R, LassoError>
where
    F: std::future::Future<Output = Result<R, LassoError>>,
{
    let start = Instant::now();
    let result = f.await;
    let duration = start.elapsed();
    tracker.record_step(name, duration);
    result
}
