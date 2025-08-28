"""
Flywheel Analysis Download Script - Recon-All-Clinical with Gambas Filter

This script downloads the most recent analysis results from Flywheel for specified
gear runs across multiple projects and groups. It handles duplicate analyses by
selecting only the most recent run for each subject-session combination.
Modified for recon-all-clinical gear with gambas filter.

Author: niall.bourke@kcl.ac.uk
Date: 08/07/2025
"""

import os
import flywheel
from pathlib import Path
import pathvalidate as pv
import pandas as pd
from datetime import datetime
import pytz
import time
import logging
import subprocess

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_output_directory(base_path, project_label, gear_name,job_name_filter):
    """
    Create and return the output directory path for a given project and gear.
    
    Args:
        base_path (Path): Base directory for all outputs
        project_label (str): Project label from Flywheel
        gear_name (str): Name of the gear being processed
        
    Returns:
        Path: Sanitized path to the project-specific output directory
    """
    project_path = pv.sanitize_filepath(base_path / project_label / gear_name / job_name_filter, platform='auto')
    if not project_path.exists():
        project_path.mkdir(parents=True)
        logger.info(f"Created output directory: {project_path}")
    return project_path

def find_most_recent_analysis(session, gear_name, gear_versions, timestamp_filter, job_name_filter=None):
    """
    Find the most recent analysis for a session that matches the specified criteria.
    
    Args:
        session: Flywheel session object
        gear_name (str): Name of the gear to search for
        gear_versions (list): List of acceptable gear versions
        timestamp_filter (datetime): Only include analyses after this date
        job_name_filter (str): Optional string that must be in the job name
        
    Returns:
        analysis object or None: Most recent matching analysis, or None if no matches
    """
    matching_analyses = []
    
    for analysis in session.analyses:
        if (analysis.gear_info is not None and 
            analysis.gear_info.name == gear_name and 
            analysis.gear_info.version in gear_versions and 
            analysis.created > timestamp_filter):
            
            # Check if job name filter is specified and if it matches
            if job_name_filter:
                # Check both job.name and label for the filter string
                job_name = getattr(analysis.job, 'name', '') if hasattr(analysis, 'job') and analysis.job else ''
                analysis_label = getattr(analysis, 'label', '')
                
                if job_name_filter.lower() not in job_name.lower() and job_name_filter.lower() not in analysis_label.lower():
                    logger.debug(f"Skipping analysis - job name '{job_name}' and label '{analysis_label}' do not contain '{job_name_filter}'")
                    continue
                
                logger.info(f"Found matching analysis with job name: '{job_name}' or label: '{analysis_label}'")
            
            matching_analyses.append(analysis)
    
    if matching_analyses:
        # Sort by creation date (most recent first) and return the first one
        most_recent = sorted(matching_analyses, key=lambda x: x.created, reverse=True)[0]
        logger.info(f"Found {len(matching_analyses)} matching analyses, "
                   f"using most recent from {most_recent.created} "
                   f"(version {most_recent.gear_info.version})")
        return most_recent
    
    return None

def download_analysis_files(analysis, output_dir, subject_label, session_label, file_pattern="volume"):
    """
    Download files from an analysis that match the specified pattern.
    
    Args:
        analysis: Flywheel analysis object
        output_dir (Path): Directory to save downloaded files
        subject_label (str): Subject identifier
        session_label (str): Session identifier
        file_pattern (str): Pattern to match in filename (default: "volume")
        
    Returns:
        list: List of DataFrames containing the downloaded CSV data
    """
    dataframes = []
    
    for analysis_file in analysis.files:
        if file_pattern in analysis_file.name:
            file = analysis_file.reload()
            
            # Create download directory structure
            download_dir = pv.sanitize_filepath(output_dir / subject_label / session_label, platform='auto')
            if not download_dir.exists():
                download_dir.mkdir(parents=True)
            
            # Set up download path
            download_path = download_dir / file.name
            logger.info(f"Downloading: {download_path}")
            
            # Download the file
            file.download(download_path)
            logger.info(f"Downloaded file for {session_label}, analysis: {analysis.label}")
            
            # Read CSV and add to dataframes list
            try:
                results = pd.read_csv(download_path, index_col=None, header=0)
                dataframes.append(results)
                logger.info(f"Successfully loaded CSV with {len(results)} rows")
            except Exception as e:
                logger.error(f"Failed to read CSV {download_path}: {str(e)}")
    
    return dataframes

def sanitize_filename(project_label):
    """
    Sanitize project label for use in filename.
    
    Args:
        project_label (str): Raw project label
        
    Returns:
        str: Sanitized filename-safe string
    """
    sanitized = project_label.replace(' ', '_')
    sanitized = sanitized.replace('(', '')
    sanitized = sanitized.replace(')', '')
    return sanitized

def main():
    """Main execution function."""
    # Record start time
    start_time = time.time()
    logger.info("Starting Flywheel recon-all-clinical analysis download script")
    
    # =============================================================================
    # CONFIGURATION SECTION
    # =============================================================================
    
    # Flywheel connection
    # Source zshrc and get the env var you want
    def get_env_from_zshrc(var_name):
        command = f"source ~/.zshrc && echo ${var_name}"
        result = subprocess.run(['zsh', '-c', command], stdout=subprocess.PIPE, text=True)
        return result.stdout.strip()

    api_key = get_env_from_zshrc('FW_CLI_API_KEY')
    # api_key = os.environ.get('FW_CLI_API_KEY')
    if not api_key:
        raise ValueError("FW_CLI_API_KEY environment variable not set")
    
    fw = flywheel.Client(api_key=api_key)
    
    # Groups to process
    GROUP_NAMES = ["global_map", "prisma"]
    
    # Analysis filters
    GEAR_NAME = 'recon-all-clinical'  # Changed from 'minimorph'
    GEAR_VERSIONS = ['0.4.3', '0.4.4', '0.4.7']  # Changed to recon-all-clinical version
    TIMESTAMP_FILTER = datetime(2025, 6, 1, 0, 0, 0, 0, pytz.UTC)  # Only analyses after this date
    JOB_NAME_FILTER = 'mrr-axireg' #'gambas'  # Must be in job name
    
    # Projects to process (leave empty list to process all projects)
    TARGET_PROJECTS = [ 
        "Botswana-MOTHEO"]
    
    
    # Output configuration
    BASE_WORK_DIR = Path.home() / 'unity/fw-notebooks/Data'
    FILE_PATTERN = "volume"  # Pattern to match volume.csv files
    
    # =============================================================================
    # MAIN PROCESSING LOOP
    # =============================================================================
    
    # Create base work directory
    if not BASE_WORK_DIR.exists():
        BASE_WORK_DIR.mkdir(parents=True)
        logger.info(f"Created base work directory: {BASE_WORK_DIR}")
    
    # Process each group
    for group_name in GROUP_NAMES:
        logger.info(f"Processing group: {group_name}")
        
        try:
            group = fw.lookup(group_name)
            projects = group.projects()
            
            # Process each project in the group
            for project in projects:
                # Skip projects not in target list (if target list is specified)
                if TARGET_PROJECTS and project.label not in TARGET_PROJECTS:
                    continue
                
                logger.info(f"Processing project: {project.label}")
                
                # Set up output directory for this project
                project_output_dir = create_output_directory(BASE_WORK_DIR, project.label, GEAR_NAME, JOB_NAME_FILTER)
                
                # Collect all dataframes and version info for this project
                all_dataframes = []
                versions_used = set()  # Track actual versions used
                
                # Process each subject in the project
                for subject in project.subjects.iter():
                    subject = subject.reload()
                    subject_label = subject.label
                    
                    # Process each session for this subject
                    for session in subject.sessions.iter():
                        session = session.reload()
                        session_label = session.label
                        
                        logger.info(f"Processing: {subject_label} - {session_label}")
                        
                        # Find the most recent matching analysis with gambas filter
                        most_recent_analysis = find_most_recent_analysis(
                            session, GEAR_NAME, GEAR_VERSIONS, TIMESTAMP_FILTER, JOB_NAME_FILTER
                        )
                        
                        if most_recent_analysis:
                            analysis_version = most_recent_analysis.gear_info.version
                            versions_used.add(analysis_version)  # Track the version used
                            logger.info(f"Found analysis: {GEAR_NAME} v{analysis_version} with '{JOB_NAME_FILTER}' in job name")
                            
                            # Download and process files from this analysis
                            session_dataframes = download_analysis_files(
                                most_recent_analysis, 
                                project_output_dir, 
                                subject_label, 
                                session_label, 
                                FILE_PATTERN
                            )
                            
                            all_dataframes.extend(session_dataframes)
                        else:
                            logger.info(f"No matching analyses found for {subject_label} - {session_label} with '{JOB_NAME_FILTER}' in job name")
                
                # =============================================================================
                # SAVE COMBINED RESULTS
                # =============================================================================
                
                if all_dataframes:
                    try:
                        # Create output filename with actual versions used
                        sanitized_project_name = sanitize_filename(project.label)
                        
                        # Create version string from actual versions used
                        if versions_used:
                            versions_str = "-".join(sorted(versions_used))
                            output_filename = f"{sanitized_project_name}-{GEAR_NAME}-v{versions_str}-{JOB_NAME_FILTER}-volumes.csv"
                        else:
                            # Fallback if no versions tracked
                            output_filename = f"{sanitized_project_name}-{GEAR_NAME}-{JOB_NAME_FILTER}-volumes.csv"
                        
                        # Combine all dataframes
                        combined_df = pd.concat(all_dataframes, axis=0, ignore_index=True)
                        output_path = project_output_dir / output_filename
                        
                        # Save to CSV
                        combined_df.to_csv(output_path, index=False)
                        logger.info(f"Saved combined results to: {output_path}")
                        logger.info(f"Total rows in combined dataset: {len(combined_df)}")
                        logger.info(f"Gear versions used: {sorted(versions_used)}")
                        
                        # Upload to Flywheel project
                        project.upload_file(str(output_path))
                        logger.info(f"Uploaded results to Flywheel project: {project.label}")
                        
                    except Exception as e:
                        logger.error(f"Failed to save/upload results for {project.label}: {str(e)}")
                        logger.error("Check that analyses were run and results are available")
                        continue
                else:
                    logger.warning(f"No data found for project: {project.label}")
        
        except Exception as e:
            logger.error(f"Failed to process group {group_name}: {str(e)}")
            continue
    
    # =============================================================================
    # COMPLETION
    # =============================================================================
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"Script completed successfully in {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()