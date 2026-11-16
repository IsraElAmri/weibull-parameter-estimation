# Backblaze Hard Drive Test Data — Q3 2025

## Source
- **Provider**: Backblaze Inc.
- **URL**: https://www.backblaze.com/cloud-storage/resources/hard-drive-test-data
- **Quarter**: Q3 2025 (July 1 – September 30, 2025)

## Format
- **Files**: One CSV per day (92 files total)
- **Size**: ~127 MB per file, ~11.7 GB total

## Key Columns
| Column | Description |
|--------|-------------|
| `date` | Observation date (YYYY-MM-DD) |
| `serial_number` | Unique drive identifier |
| `model` | Drive model name |
| `capacity_bytes` | Drive capacity in bytes |
| `failure` | 0 = operational, 1 = failed on this day |
| `smart_*_raw` | SMART attribute raw values |

## Important SMART Attributes
| Attribute | Meaning |
|-----------|---------|
| `smart_5_raw` | Reallocated sector count |
| `smart_9_raw` | Power-on hours (age proxy) |
| `smart_187_raw` | Reported uncorrectable errors |
| `smart_188_raw` | Command timeout count |
| `smart_197_raw` | Current pending sector count |
| `smart_198_raw` | Offline uncorrectable sector count |

## Data Placement
Place all daily CSV files (e.g., `2025-07-01.csv`, ..., `2025-09-30.csv`)
in this `data/` directory, **or** update `DATA_RAW_DIR` in `R/01_data_prep.R`
to point to the directory containing the raw CSV files.

## Right-Censoring
- `failure == 1`: Drive failed — exact failure time observed.
- `failure == 0`: Drive survived through the observation window — 
  right-censored. True failure time is unknown but at least as long
  as the observation period.
