# assignment-fs

Slides:
https://docs.google.com/presentation/d/1L6se6_fbHI7norV-QgWYeJO89GFDdNo64ERgL1WbPJc/edit?usp=sharing

Steam Games Data Processing Pipeline - README
Overview
This repository contains a comprehensive data processing pipeline for Steam games data, consisting of two main Python scripts that clean, transform, and analyze a large-scale dataset of 140,082 Steam games with associated metadata from 8 different data sources.
Dataset Information
Input Files Required
The pipeline expects the following CSV files in the directory /Users/timtoepper/Downloads/Banks_Assignment_Tim/:

games.csv or games_cleaned.csv (140,082 records)

Core game information including app_id, name, type, release date, pricing


steamspy_insights.csv (140,077 records)

User engagement metrics, ownership data, playtime statistics


reviews.csv (140,086 records)

Review scores, positive/negative counts, metacritic scores, review text


categories.csv (522,582 records)

Game categories (multiplayer, single-player, co-op, VR support, etc.)


tags.csv (1,744,632 records)

User-generated tags for games (average 14.8 tags per game)


genres.csv (353,339 records)

Official genre classifications


descriptions.csv (59,858 records)

Game descriptions in multiple formats (summary, extensive, detailed)


promotional.csv (10,558 records)

Promotional materials metadata



Total Dataset Size

Primary records: 140,082 games
Total records across all files: ~2.7 million
Final merged dataset: 140,082 rows × 107 columns

Scripts Description
Script 1: games_preprocessing.py
Purpose: Initial preprocessing and sampling of games data
Key Features:

Parses JSON price_overview field into structured columns
Cleans HTML tags from language fields
Detects full audio language support
Creates both full cleaned dataset and 10,000-record sample
Implements block-based sampling for representative subset

Processing Time: ~11 seconds
Output Files:

sampled_outputs/games_cleaned.csv (140,082 rows × 14 columns)
sampled_outputs/games_sampled.csv (10,000 rows × 14 columns)

Script 2: complete_data_pipeline.py
Purpose: Comprehensive data integration, cleaning, and M&A analysis
Key Features:

Merges all 8 data sources
Currency conversion to USD (supports 40+ currencies)
Text cleaning for reviews and descriptions
Advanced KPI calculations
Publisher-level aggregation
M&A target scoring and analysis

Processing Time: ~2-3 minutes
Output Files:

output/steam_data_cleaned_complete.csv (140,082 rows × 107 columns)
output/steam_data_cleaned_sample_1000.csv (1,000 row sample)
output/publisher_metrics_cleaned.csv (47,307 publishers)
output/publisher_ma_analysis.csv (M&A scoring for publishers)
output/publisher_metrics_top100.csv (Top 100 publishers)
output/data_quality_report.txt

Installation & Requirements
Python Version
Python 3.8 or higher
Required Libraries
bashpip install pandas numpy pathlib
Directory Structure
/Users/timtoepper/Downloads/Banks_Assignment_Tim/
├── games.csv (or games_cleaned.csv)
├── steamspy_insights.csv
├── reviews.csv
├── categories.csv
├── tags.csv
├── genres.csv
├── descriptions.csv
├── promotional.csv
├── sampled_outputs/          # Created by Script 1
│   ├── games_cleaned.csv
│   └── games_sampled.csv
└── output/                   # Created by Script 2
    ├── steam_data_cleaned_complete.csv
    ├── steam_data_cleaned_sample_1000.csv
    ├── publisher_metrics_cleaned.csv
    ├── publisher_ma_analysis.csv
    ├── publisher_metrics_top100.csv
    └── data_quality_report.txt
    
Usage Instructions
Step 1: Run Initial Preprocessing
pythonpython games_preprocessing.py
This creates cleaned games data in sampled_outputs/ directory.
Step 2: Run Complete Pipeline
pythonpython complete_data_pipeline.py
This performs full data integration and analysis.
Data Cleaning Operations
Currency Handling

Normalizes 40+ currency codes
Converts all prices to USD using fixed exchange rates
Handles missing/malformed currency data (defaults to USD)
Processes both cents and dollar formats

Text Cleaning

Removes HTML tags (<br>, <strong>, etc.)
Handles HTML entities (&quot;, &amp;, etc.)
Strips escape characters (\n, \r, \t)
Removes backslash escaping
Normalizes whitespace

Missing Data Handling

Treats \N, \\N, N, empty strings as NULL
Applies appropriate defaults (0 for numeric, empty for text)
Preserves data quality metrics

Key Metrics Calculated
Game-Level Metrics

Estimated Revenue (USD): owners × average price
Game Age: years since release
Success Indicators: based on review thresholds
Quality Score: positive review ratio × 100
Hit Status: top 10% by ownership

Publisher-Level Metrics

Portfolio Size: number of games
Total User Base: sum of game owners
Revenue Metrics: total and per-game revenue
Success Rates: commercial and critical
Content Production: release cadence, recent games ratio

M&A Scoring Components

Content Score (25%): Release cadence, portfolio size
Quality Score (25%): Hit rate, success rates
Efficiency Score (20%): ROI, revenue per user
Strategic Fit Score (20%): Retention, franchises, market reach
Risk Adjustment (10%): Revenue concentration, platform dependency

Data Quality Statistics
Overall Completeness

Data Completeness: 84.6%
Total Revenue Tracked: $229.6 billion USD
Valid Publisher Data: 90,368 games (64.5%)

Key Field Coverage

Price Data: 100% (with USD conversion)
Owner Data: 99.9%
Review Data: 35.2%
Description Data: 42.7%
Tag Data: 100%

Performance Optimization
Memory Management

Vectorized operations for publisher aggregations
Chunked reading for large CSV files
Efficient dtype conversions

Processing Speed

Script 1: ~11 seconds for 140K games
Script 2: ~2-3 minutes for full pipeline
Handles malformed CSV data gracefully

Advanced Features
Franchise Detection

Identifies sequels via naming patterns
Tracks sequel performance vs originals
Calculates franchise ratios

Content Analysis

Mature content detection
Family-friendly classification
Technology usage (VR, ray tracing, retro)
Platform exclusivity tracking

Risk Assessment

Revenue concentration (Herfindahl Index)
Platform dependency metrics
Controversy detection from reviews
"Zombie game" identification

Output File Descriptions
steam_data_cleaned_complete.csv
Complete merged dataset with all 107 columns including:

Core identifiers and metadata
USD-converted pricing
User engagement metrics
Review scores and text
Categories, tags, and genres
Clean descriptions
Calculated KPIs

publisher_metrics_cleaned.csv
Publisher-level aggregations with:

Portfolio statistics
Revenue metrics (USD)
Success rates
User base totals

publisher_ma_analysis.csv
Advanced M&A metrics including:

M&A scores (0-100)
Acquisition tier classifications
Component scores (content, quality, efficiency, strategic fit)
Risk metrics

Error Handling
Malformed Data

Handles CSV parsing errors gracefully
Skips bad lines with logging
Continues processing on partial failures

Missing Files

Checks for alternative file names
Provides clear error messages
Continues with available data

Notes and Limitations

Exchange Rates: Fixed rates as of 2025-08-01
Revenue Estimates: Based on ownership × price (approximate)
Sampling: Block-based sampling may not perfectly represent tail distribution
Text Processing: Some edge cases in HTML cleaning may remain

Support and Troubleshooting
Common Issues

Memory Errors: Reduce sample size or process in chunks
File Not Found: Verify all 8 input CSVs are present
Encoding Issues: Script handles UTF-8 with error ignoring

Performance Tips

Use sampled data for testing (games_sampled.csv)
Run in the cloud

Dataset Statistics Summary

Total Games: 140,082
Total Publishers: 47,307
Total Revenue (USD): $229.6 billion
Average Tags per Game: 14.8
Commercial Successes: 32,456 games
Critical Successes: 17,526 games
Hidden Gems: 293 games
Dead Games: 89,814 games

License and Attribution
This pipeline processes publicly available Steam data. Ensure compliance with Steam's terms of service and data usage policies when using the processed outputs.

Last Updated: August 2025
Pipeline Version: 2.0 (Enhanced with M&A Analytics)
