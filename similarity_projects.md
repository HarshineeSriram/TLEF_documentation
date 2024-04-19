# Project Summary and Embedding Documentation

This documentation explains the logic that processes project summaries, generate embeddings, and identify similar projects based on these embeddings. The script primarily uses `pandas` for data manipulation, `boto3` for AWS interactions, `SentenceTransformer` for generating embeddings, and `numpy` for numerical operations.

## Script Overview

### Import Libraries
The script begins by importing necessary libraries, including:
- `boto3`: Amazon Web Services (AWS) SDK for Python.
- `pandas` and `numpy`: For data handling and mathematical operations.
- `pickle`: For object serialization.
- `os` and `sys`: For interacting with the operating system and the system-specific parameters.
- `scipy.spatial.distance.cdist`: For computing distances between point sets.

### Configuration Parameters
- Retrieves configuration parameters such as bucket names and file paths from AWS Glue's `getResolvedOptions` function, which extracts these parameters from the command line.

### Helper Functions
- `createDir`: Creates a directory and returns its path.
- `return_df`: Loads data from a specified bucket into a DataFrame based on the file type.

### Core Functions
- `find_all_summaries`: Iterates over projects, generates embeddings for project summaries, and stores them.
- `generate_context_embeddings`: Generates concatenated embeddings for projects based on their titles and summaries.
- `check_and_update_embeddings`: Checks existing embeddings and updates them if the similarity with new data is below a threshold.
- `store_context_and_embeddings`: Stores project context and embeddings in S3.
- `generate_embeddings_database`: Creates a database of embeddings from S3 data.
- `generate_similar_projects_database`: Identifies similar projects based on embeddings.
- `save_similar_projects_database`: Saves the similar projects database to an S3 bucket.

### Execution Flow
- The `main` function orchestrates the execution by calling the embedding generation and similarity calculation functions, followed by saving the output.

### Detailed Execution Flow

The `main` function in this script serves as the entry point and orchestrates the processing and analysis of project data using a series of function calls. Below, the detailed step-by-step execution flow is outlined, highlighting the sequence of function invocations, their inputs, and outputs.

## Main Function (`main`)

1. **Initialize the process by calling `find_all_summaries`.**
   - No explicit parameters are passed to this function; it utilizes global variables defined from the AWS Glue parameters.
   - **Purpose**: To iterate through each project described in a DataFrame, generate embeddings for the project summaries, and store these embeddings in the specified S3 bucket.

2. **Generate embeddings database by invoking `generate_embeddings_database` with `EMBEDDINGS_BUCKET` as an argument.**
   - **Input**: `EMBEDDINGS_BUCKET` (string) - The name of the S3 bucket where embeddings are stored.
   - **Output**: `embeddings_database` (dictionary) - A dictionary where keys are the object names in S3 and values are their corresponding embeddings.
   - **Purpose**: To retrieve all existing embeddings from S3 and compile them into a local dictionary for further processing.

3. **Generate similar projects database by calling `generate_similar_projects_database` with the `EMBEDDINGS_BUCKET` and `embeddings_database` as arguments.**
   - **Input**: 
     - `EMBEDDINGS_BUCKET` (string) - The S3 bucket name.
     - `embeddings_database` (dictionary) - The dictionary of embeddings previously generated.
   - **Output**: `similar_projects_database` (dictionary) - A dictionary mapping each project to a list of similar projects based on embedding similarity.
   - **Purpose**: To compute similarity scores between all project embeddings and identify the most similar projects for each.

4. **Save the similar projects database by calling `save_similar_projects_database` with `similar_projects_database` and a specified save path as arguments.**
   - **Input**:
     - `similar_projects_database` (dictionary) - The dictionary containing each project and its similar projects.
     - `save_path` (string) - The S3 URI where the similar projects data will be saved, formatted as a parquet file.
   - **Purpose**: To save the similar projects information back to S3, allowing for later retrieval and analysis.

## Subsequent Function Calls and Data Flow

- **`find_all_summaries`**: Iterates over project IDs from a DataFrame and calls `store_context_and_embeddings` for each project.
  - **Intermediate Calls**:
    - `generate_context_embeddings` might be called within `find_all_summaries` if multiple entries for a project need to be processed together to generate a comprehensive context and embedding.
  
- **`store_context_and_embeddings`**:
  - First attempts to update existing embeddings by calling `check_and_update_embeddings`.
  - If there are no existing embeddings for this project, the currently generated embeddings are directly stored as new data in S3.
  - Note: If there are existing embeddings, `check_and_update_embeddings` has already taken care of them.
  - **Inputs**: Project context and embeddings, the bucket to store in, and the data key for storage.
  
- **`generate_context_embeddings`**:
  - Processes multiple entries for a project to create a combined context string and its embedding.
  - **Outputs**: A tuple containing the combined context and its embedding.

### S3 Interactions
- The script uses `boto3` client methods to interact with AWS S3 for data storage and retrieval.

### Embedding Model
- Utilizes `SentenceTransformer` for generating semantic embeddings of project summaries.

## Detailed Function Descriptions

### `createDir`
```python
def createDir(path):
    os.makedirs(path, exist_ok=True)
    return os.path.join(os.getcwd(), path)
```
Creates a directory for storing cache files, with permissions to create intermediate directories if necessary.

### `return_df`
```python
def return_df(bucket, data_key):
    if "s3://" in data_key:
        data_location = data_key
    else:
        data_location = f's3://{bucket}/{data_key}'
    if ".parquet" in data_key:
        df = pd.read_parquet(data_location)
    elif ".xlsx" in data_key:
        df = pd.read_excel(data_location)
    elif ".csv" in data_key:
        df = pd.read_csv(data_location)
    return df
```
Loads data from AWS S3 based on the file type and returns it as a pandas DataFrame.

### `find_all_summaries`
Iterates through each project ID, processes the relevant data, and invokes embedding storage functions.

### `check_and_update_embeddings`
Compares new embeddings with existing ones in S3, and updates them if the cosine similarity is below a specified threshold.

### `store_context_and_embeddings`
Stores or updates the context and embeddings in AWS S3, ensuring updates only occur when necessary.

### `generate_context_embeddings`
Concatenates and encodes context from project titles and summaries, updating embeddings if the new content provides additional context.

### Usage
To run this script, ensure AWS CLI is configured correctly with the necessary permissions, and the required libraries are installed in your Python environment. This script is typically executed in an AWS environment where AWS Glue jobs are configured. Adjust the parameters and paths according to your AWS setup.
