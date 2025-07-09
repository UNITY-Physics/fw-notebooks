import os
import flywheel
from pathlib import Path
import pathvalidate as pv
import pandas as pd
from datetime import datetime
import pytz
import time
import argparse

"""   
Pull Results from Flywheel
Specify the keyword (to look for in the file of interest), project name, gear, and it version

"""

# get the start time
st = time.time()

# Flywheel connector
# Extract gear and gear version from cmd arguments
parser = argparse.ArgumentParser(prog='pull FW results')
parser.add_argument('--apikey','-apikey', nargs='?', help='FW CLI API key')
parser.add_argument('--gear','-gear', nargs='?', help='gear name')
parser.add_argument('--gearV', '-gearV',nargs='?', help='gear version')
parser.add_argument('--keyword', '-keyword', nargs='?', help='keyword in filename')
parser.add_argument('--debug', '-debug', nargs='?', default=0, help='keyword in filename')


args = parser.parse_args()


api_key = args.apikey
gear = (str(args.gear)).strip() # recode this as variable that user selects in config
gearVersion = (str(args.gearV)).strip()
keyword = (str(args.keyword)).strip()
debug = bool(str(args.debug).strip())


group_names = ["global_map","prisma"]
project_labels = []
fw = flywheel.Client(api_key=api_key)


for group_name in group_names:    
    # Get the group
    group = fw.lookup(f"{group_name}")
    group = group.reload()
    # Get the projects in the group
    projects = group.projects()
    project_labels.extend([project.label for project in projects])



timestampFilter = datetime(2024, 7, 10, 0, 0, 0, 0, pytz.UTC) # Date before which we want to filter analyses (i.e. only get analyses run after this date)



download_path = Path.cwd() / 'tmp'
download_path.mkdir(parents=True, exist_ok=True)

if debug:
    projects = [projects[0]]
# Loop over all projects
for project_label in projects:   
    
        print(f"Processing: {project_label}")
        project = fw.projects.find_first(f"label={project_label}").reload()
        # --- prep output --- #

        
        # Create a custom path for our project (we may run this on other projects in the future) and create if it doesn't exist
        project_path = download_path / project.label / gear
        project_path = pv.sanitize_filepath(project_path) 

        if not project_path.exists():
            project_path.mkdir(parents = True)
        # Preallocate lists
        df = []
        
        # --- Find the results --- #

        # Iterate through all subjects in the project and find the results
        for subject in project.subjects.iter():
            subject = subject.reload()
            sub_label = subject.label
            for session in subject.sessions.iter():
                session = session.reload()
                ses_label = session.label

                print(sub_label, ses_label)
                for analysis in session.analyses:
                    #print("Analyses ran on this subject: ", analysis.gear_info.name, analysis.gear_info.version)
                    if analysis.gear_info is not None and analysis.gear_info.name == gear and analysis.created > timestampFilter and analysis.gear_info.version == gearVersion and analysis.get("job").get("state") == "complete":
                        print("pulling: ", gear, gearVersion)
                        for analysis_file in analysis.files:
                            if keyword in analysis_file.name:
                                file = analysis_file
                                file = file.reload()
                                # Sanitize our filename and parent path
                                download_dir = pv.sanitize_filepath(project_path/sub_label/ses_label,platform='auto')   
                                fileName = file.name #(analysis.gear_info.name + "_" + analysis.label + ".csv")

                                # Create the path
                                if not download_dir.exists():
                                    download_dir.mkdir(parents=True)
                                download_path = download_dir/fileName
                                

                                # Download the file
                                print('Downloading file: ', ses_label, analysis.label)
                                file.download(download_path)

                                # Add subject to dataframe
                                with open(download_path) as csv_file:
                                    results = pd.read_csv(csv_file, index_col=None, header=0) 
                                    df.append(results)
                                    
        # --- Save output --- #

        try:
            outname = project.label.replace(' ', '_')
            outname = outname.replace('(', '')
            outname = outname.replace(')', '')
            filename = (outname + "-" + gear + "-" + gearVersion + "-" + f"{keyword}.csv")
            # write DataFrame to an excel sheet 
            df = pd.concat(df, axis=0, ignore_index=True)
            outdir = os.path.join(project_path, filename)
            df.to_csv(outdir)

            # UPLOADS A FILE TO THE PROJECT INFORMATION TAB
            project.upload_file(outdir)
        except:
            print(f"Failed to upload results for {project.label} to project info tab")
            print("Check that if analysis was run, and that there are results to upload")
            continue

# get the end time
et = time.time()
# get the execution time
elapsed_time = et - st
print('Execution time:', elapsed_time, 'seconds')

