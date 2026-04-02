# Data Processing for LLM Competition

## Overview
This repository contains scripts used to improve model performance through data design.

## Data Processing

This script was used to improve model performance through data design.

Key ideas:

- Normalize different datasets into consistent formats  
  (ALFWorld: action-only, DBBench: SQL-only)

- Identify weak task types (e.g., aggregation, counting)  
  and upweight them by duplication

- Combine and shuffle datasets to control distribution

These steps were based on the hypothesis that data quality and distribution
significantly affect model performance.

## Code
See `data_processing.py`
