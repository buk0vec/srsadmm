# srsadmm: Serverless Rust ADMM

## Overview

This is a Rust implementation of the Alternating Direction Method of Multipliers (ADMM) algorithm. It is designed to be used in a serverless environment, where the heavy computation (generally matrix multiplications) is offloaded to serverless functions.

## Structure

- `srsadmm-core` is the core library containing the ADMM algorithm, as well as binaries to solve a LASSO regression problem.
- `srsadmm-lambda-mm` is the serverless matrix multiplication function.
- `srsadmm-lasso-factor` is a serverless function that computes (A^T A + lambda I) for a LASSO regression problem (not used for this implementation).

## Usage

- Deploy the `srsadmm-lambda-mm` function to AWS Lambda using `cargo lambda build --release` followed by `cargo lambda deploy`.
- Within the `srsadmm-core` library, run the `generate_problem` binary to generate a problem instance.
- Run the `lasso` binary to solve the problem instance, or `lasso_prox` to solve the problem instance with the proximal gradient method.

## License

Please give me credit if you decide to copy this code for some reason.