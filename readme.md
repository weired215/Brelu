# Brelu

## Overview

This script provides three different operation modes for BReLU  functionality, allowing you to perform fault injection attacks, boundary discovery, or model protection.

## Usage Syntax

```bash
# Basic fault injection attack (default mode)
bash brelu.sh

# Find and save BReLU boundaries using training data
bash brelu.sh --enable-find-boundaries

# Protect model with BReLU and verify protection
bash brelu.sh --enable-fixed
```

## Operation Modes

### 1. Default Mode: Fault Injection Attack

**Command:** `bash brelu.sh`

**Description:**

Performs Bit-Flip Attack (BFA) on the target model without any BReLU boundary modifications. This mode uses the original model weights and injects faults to test model vulnerability.

### 2. Boundary Discovery Mode

**Command:** `bash brelu.sh --enable-find-boundaries`

**Description:**

Utilizes training data to automatically discover optimal boundaries for BReLU activation function. The discovered boundaries are saved for future use in model protection.

### 3. Model Protection Mode

**Command:** `bash brelu.sh --enable-fixed`

**Description:**

Replaces standard ReLU with BReLU using pre-discovered fixed boundaries to protect the model against fault injection attacks. Validates the protection effectiveness.



## environment

refer the requirement.txt