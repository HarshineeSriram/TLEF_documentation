# Preparing the dataset for semantic search to determine similar projects <a name="similar-projects-data-cleaning-documentation"></a>

## Table of Contents <a name="table-of-contents"></a> 
1. [Script Overview](#script-overview)
2. [Configuration Parameters](#configuration-parameters)
3. [Function Descriptions](#function-descriptions)
   - [return_df](#return-df)
   - [store_df](#store-df)
   - [assign_grant_ids](#assign-grant-ids)
   - [convert_summary_to_strings](#convert-summary-to-strings)
4. [Main Script Execution](#main-script-execution)
5. [Glue Job Trigger](#glue-job-trigger)

## Script Overview <a name="script-overview"></a>
This script handles the generation of unique grant IDs and the initiation of the Glue job to generate the similar projects database for further data processing.

## Configuration Parameters <a name="configuration-parameters"></a>
The script retrieves configuration parameters using the `getResolvedOptions` function from AWS Glue utilities, which include:
- `BUCKET_NAME`: The main bucket for storing project details.
- `PROJECT_DETAILS_S3URI`: The S3 URI for the project details Excel file.
- `EMBEDDINGS_BUCKET`: The bucket for storing embeddings.

```python
from awsglue.utils import getResolvedOptions
args = getResolvedOptions(sys.argv, ["BUCKET_NAME", "PROJECT_DETAILS_S3URI", "EMBEDDINGS_BUCKET"])
```

## Function Descriptions <a name="function-descriptions"></a>
### `return_df` <a name="return-df"></a>
This function loads data from a specified S3 bucket into a pandas DataFrame. It accepts the bucket name and data key. If the data key is a full S3 URI, it uses it directly; otherwise, it formats it as `s3://{bucket}/{data_key}`.

```python
def return_df(bucket, data_key):
    data_location = data_key if "s3://" in data_key else f's3://{bucket}/{data_key}'
    df = pd.read_excel(data_location)
    return df
```

### `store_df` <a name="store-df"></a>
Stores a DataFrame into an S3 bucket at a specified location. It formats the destination as `s3://{bucket}/{data_key}` and checks if the target file type is Parquet to call `df.to_parquet`, otherwise it uses `df.to_excel`.

```python
def store_df(df, bucket, data_key):
    data_location = f's3://{bucket}/{data_key}'
    if '.parquet' in data_key:
        df.to_parquet(data_location, index=False)
    else:
        df.to_excel(data_location, index=False)
```

### `assign_grant_ids` <a name="assign-grant-ids"></a>
Processes each project in a DataFrame to assign a unique grant ID based on several attributes such as the project faculty, type, and principal investigator's name.

```python
def assign_grant_ids(project_details_df):
    
    grant_ids = []
    
    for proj_idx in range(len(project_details_df)):
        
        this_proj_funding_year = project_details_df.funding_year.iloc[proj_idx]
        
        if ';' in project_details_df.project_faculty.iloc[proj_idx]: # There are multiple project faculties
            this_proj_faculty = ""
            
            faculty_list = project_details_df.project_faculty.iloc[proj_idx].split(';') # Add all faculties to the grant ID
            for this_faculty in faculty_list:
                this_proj_faculty += this_faculty.strip() + '-'
            
            this_proj_faculty = this_proj_faculty[:-1] # Remove the last '-'
            
        else:
            this_proj_faculty = project_details_df.project_faculty.iloc[proj_idx].strip()
        
        this_proj_type = ""
        
        existing_grant_id = project_details_df.grant_id.iloc[proj_idx]
        
        if not existing_grant_id != existing_grant_id: # Existing Grant ID is not NaN
            if len(existing_grant_id.split('-')) >= 5: # Existing Grant ID has 5 components or 6 in some cases ('2015-TLEF-LP1-ARTS-Giltrow-A' and '2015-TLEF-LP1-ARTS-Giltrow-B')

                if "SP" in existing_grant_id.split('-')[2] or "LP" in existing_grant_id.split('-')[2]:
                    this_proj_type = existing_grant_id.split('-')[2] # Extract project type from the existing Grant ID
                
                elif "FL" in existing_grant_id.split('-')[1]: # Convert it into an LP but restore the number, e.g., FL2 -> LP2
                    if existing_grant_id.split('-')[1][-1].isdigit(): # E.g., "FL2"
                        this_proj_type = "LP" + existing_grant_id.split('-')[1][-1]
                    
                    else:
                        this_proj_type = "LP"
                        
                elif "UDL" in existing_grant_id.split('-')[1]: # Convert it into an SP but restore the number, e.g., UDL2 -> SP2
                    if existing_grant_id.split('-')[1][-1].isdigit(): # E.g., "UDL2"
                        this_proj_type = "SP" + existing_grant_id.split('-')[1][-1]
                    
                    else:
                        this_proj_type = "SP"
                
                elif "ITTG" in existing_grant_id.split('-')[2]: # E.g., Grant ID = 2020-ITTG-ARTS-Mawani
                    this_proj_id = project_details_df.project_id.iloc[proj_idx] # Extract the corresponding Project ID, e.g., 2020-SP-ARTS-008
                    
                    if "SP" in this_proj_id.split('-')[1] or "LP" in this_proj_id.split('-')[1]:
                        this_proj_type = this_proj_id.split('-')[1] # Extract and store project type from the project ID
                        
        if this_proj_type == "":
            this_proj_type = "SP" if project_details_df.project_type.iloc[proj_idx] == "Small" else "LP"
        
        if ';' in project_details_df.pi_name.iloc[proj_idx]: # If there are multiple PIs
            this_proj_PI_surname = ""
            
            PI_list = project_details_df.pi_name.iloc[proj_idx].split(';')
            for PI_name in PI_list:
                this_PI_surname = PI_name.split(' ')[-1]
                this_proj_PI_surname += this_PI_surname.strip() + '-' # Add all PI surnames to the grant ID
                
            this_proj_PI_surname = this_proj_PI_surname[:-1] # Remove the last '-'
            
        else:
            this_proj_PI_surname = project_details_df.pi_name.iloc[proj_idx].split(' ')[-1].strip()

        this_proj_grant_id = str(this_proj_funding_year) + "-TLEF-" + this_proj_type + '-' + this_proj_faculty + '-' +  this_proj_PI_surname
        
        if this_proj_grant_id in grant_ids: # This PI has multiple TLEF projects in the same year
            existing_grant_id_idx = grant_ids.index(this_proj_grant_id)
            grant_ids[existing_grant_id_idx] = this_proj_grant_id + '-A'
            grant_ids.append(this_proj_grant_id + '-B')
            
        else:
            grant_ids.append(this_proj_grant_id)
    
    project_details_df["generated_grant_id"] = grant_ids
    return project_details_df
```

### `convert_summary_to_strings` <a name="convert-summary-to-strings"></a>
Converts the 'summary' column of a DataFrame to strings, handling NaN values by replacing them with empty strings. This is useful for preparing data for further text processing or analysis.

```python
def convert_summary_to_strings(project_details_df):
    
    summary_list = []
    for proj_idx in range(len(project_details_df)):
        
        this_summary = project_details_df.summary.iloc[proj_idx]
        if this_summary != this_summary: # To check if the value is NaN, in which case this will be True
            summary_list.append("")
        else:
            summary_list.append(this_summary)
    
    project_details_df.summary = summary_list
    
    return project_details_df
```

## Main Script Execution <a name="main-script-execution"></a>
The script processes an Excel file from an S3 URI to assign new grant IDs and convert summaries to strings. Then, it stores the updated DataFrame back to S3 in Parquet format.

```python
project_details_df = return_df(BUCKET_NAME, PROJECT_DETAILS_S3URI)
processed_df = convert_summary_to_strings(assign_grant_ids(project_details_df))
store_df(processed_df, BUCKET_NAME, f"staging/project_details/project_details_with_new_ids_{project_details_df['funding_year'][0]}.parquet")
```

## Glue Job Trigger <a name="glue-job-trigger"></a>
Initiates an AWS Glue job to generate embeddings and similar projects, passing necessary parameters like bucket names and project details URI.

```python
glue_client = boto3.client("glue")
glue_client.start_job_run(
    JobName="tlef-generate-embeddings-and-similar-projects",
    Arguments

={
        "--BUCKET_NAME": BUCKET_NAME,
        "--EMBEDDINGS_BUCKET": EMBEDDINGS_BUCKET,
        "--PROJECT_DETAILS_WITH_NEW_GRANT_IDS_S3URI": f"s3://{BUCKET_NAME}/{file_key}",
        "--additional-python-modules": "sentence-transformers"
    }
)
```

[🔼 Back to top](#table-of-contents)
