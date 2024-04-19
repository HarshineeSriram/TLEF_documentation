# Table of Contents
1. [Project Summary and Embedding Documentation](#project-summary-and-embedding-documentation)
   - [Script Overview](#script-overview)
     - [Import Libraries](#import-libraries)
     - [Configuration Parameters](#configuration-parameters)
     - [Helper Functions](#helper-functions)
     - [Core Functions](#core-functions)
     - [Execution Flow](#execution-flow)
     - [Detailed Execution Flow](#detailed-execution-flow)
2. [Main Function (`main`)](#main-function-main)
   - [Initial Process](#initial-process)
   - [Generate Embeddings Database](#generate-embeddings-database)
   - [Generate Similar Projects Database](#generate-similar-projects-database)
   - [Save Similar Projects Database](#save-similar-projects-database)
3. [Subsequent Function Calls and Data Flow](#subsequent-function-calls-and-data-flow)
   - [`find_all_summaries`](#find-all-summaries)
   - [`store_context_and_embeddings`](#store-context-and-embeddings)
   - [`generate_context_embeddings`](#generate-context-embeddings)
4. [S3 Interactions](#s3-interactions)
5. [Embedding Model](#embedding-model)
6. [Expanded Function Descriptions](#expanded-function-descriptions)
   - [`createDir`](#createdir)
   - [`return_df`](#return-df)
   - [`find_all_summaries`](#find-all-summaries-detailed)
   - [`check_and_update_embeddings`](#check-and-update-embeddings-detailed)
   - [`store_context_and_embeddings`](#store-context-and-embeddings-detailed)
   - [`generate_context_embeddings`](#generate-context-embeddings-detailed)

# Project Summary and Embedding Documentation <a name="project-summary-and-embedding-documentation"></a>

This documentation explains the logic that processes project summaries, generate embeddings, and identify similar projects based on these embeddings. The script primarily uses `pandas` for data manipulation, `boto3` for AWS interactions, `SentenceTransformer` for generating embeddings, and `numpy` for numerical operations.

## Script Overview <a name="script-overview"></a>

### Import Libraries <a name="import-libraries"></a>
The script begins by importing necessary libraries, including:
- `boto3`: Amazon Web Services (AWS) SDK for Python.
- `pandas` and `numpy`: For data handling and mathematical operations.
- `pickle`: For object serialization.
- `os` and `sys`: For interacting with the operating system and the system-specific parameters.
- `scipy.spatial.distance.cdist`: For computing distances between point sets.

### Configuration Parameters <a name="configuration-parameters"></a>
- Retrieves configuration parameters such as bucket names and file paths from AWS Glue's `getResolvedOptions` function, which extracts these parameters from the command line.

### Helper Functions <a name="helper-functions"></a>
- `createDir`: Creates a directory and returns its path.
- `return_df`: Loads data from a specified bucket into a DataFrame based on the file type.

### Core Functions <a name="core-functions"></a>
- `find_all_summaries`: Iterates over projects, generates embeddings for project summaries, and stores them.
- `generate_context_embeddings`: Generates concatenated embeddings for projects based on their titles and summaries.
- `check_and_update_embeddings`: Checks existing embeddings and updates them if the similarity with new data is below a threshold.
- `store_context_and_embeddings`: Stores project context and embeddings in S3.
- `generate_embeddings_database`: Creates a database of embeddings from S3 data.
- `generate_similar_projects_database`: Identifies similar projects based on embeddings.
- `save_similar_projects_database`: Saves the similar projects database to an S3 bucket.

### Execution Flow <a name="execution-flow"></a>
- The `main` function orchestrates the execution by calling the embedding generation and similarity calculation functions, followed by saving the output.

### Detailed Execution Flow <a name="detailed-execution-flow"></a>

The `main` function in this script serves as the entry point and orchestrates the processing and analysis of project data using a series of function calls. Below, the detailed step-by-step execution flow is outlined, highlighting the sequence of function invocations, their inputs, and outputs.

## Main Function (`main`) <a name="main-function-main"></a>

1. **Initialize the process by calling `find_all_summaries`.** <a name="initial-process"></a>
   - No explicit parameters are passed to this function; it utilizes global variables defined from the AWS Glue parameters.
   - **Purpose**: To iterate through each project described in a DataFrame, generate embeddings for the project summaries, and store these embeddings in the specified S3 bucket.

2. **Generate embeddings database by invoking `generate_embeddings_database` with `EMBEDDINGS_BUCKET` as an argument.** <a name="generate-embeddings-database"></a>
   - **Input**: `EMBEDDINGS_BUCKET` (string) - The name of the S3 bucket where embeddings are stored.
   - **Output**: `embeddings_database` (dictionary) - A dictionary where keys are the object names in S3 and values are their corresponding embeddings.
   - **Purpose**: To retrieve all existing embeddings from S3 and compile them into a local dictionary for further processing.

3. **Generate similar projects database by calling `generate_similar_projects_database` with the `EMBEDDINGS_BUCKET` and `embeddings_database` as arguments.** <a name="generate-similar-projects-database"></a>
   - **Input**: 
     - `EMBEDDINGS_BUCKET` (string) - The S3 bucket name.
     - `embeddings_database` (dictionary) - The dictionary of embeddings previously generated.
   - **Output**: `similar_projects_database` (dictionary) - A dictionary mapping each project to a list of similar projects based on embedding similarity.
   - **Purpose**: To compute similarity scores between all project embeddings and identify the most similar projects for each.

4. **Save the similar projects database by calling `save_similar_projects_database` with `similar_projects_database` and a specified save path as arguments.** <a name="save-similar-projects-database"></a>
   - **Input**:
     - `similar_projects_database` (dictionary) - The dictionary containing each project and its similar projects.
     - `save_path` (string) - The S3 URI where the similar projects data will be saved, formatted as a parquet file.
   - **Purpose**: To save the similar projects information back to S3, allowing for later retrieval and analysis.

## Subsequent Function Calls and Data Flow <a name="subsequent-function-calls-and-data-flow"></a>

- **`find_all_summaries`** <a name="find-all-summaries"></a>: Iterates over project IDs from a DataFrame and calls `store_context_and_embeddings` for each project.
  - **Intermediate Calls**:
    - `generate_context_embeddings` might be called within `find_all_summaries` if multiple entries for a project need to be processed together to generate a comprehensive context and embedding.
  
- **`store_context_and_embeddings`** <a name="store-context-and-embeddings"></a>:
  - First attempts to update existing embeddings by calling `check_and_update_embeddings`.
  - If there are no existing embeddings for this project, the currently generated embeddings are directly stored as new data in S3.
  - Note: If there are existing embeddings, `check_and_update_embeddings` has already taken care of them.
  - **Inputs**: Project context and embeddings, the bucket to store in, and the data key for storage.
  
- **`generate_context_embeddings`** <a name="generate-context-embeddings"></a>:
  - Processes multiple entries for a project to create a combined context string and its embedding.
  - **Outputs**: A tuple containing the combined context and its embedding.

### S3 Interactions <a name="s3-interactions"></a>
- The script uses `boto3` client methods to interact with AWS S3 for data storage and retrieval.

### Embedding Model <a name="embedding-model"></a>
- Utilizes `SentenceTransformer` for generating semantic embeddings of project summaries.

## Expanded Function Descriptions <a name="expanded-function-descriptions"></a>

### `createDir` <a name="createdir"></a>
```python
def createDir(path):
    os.makedirs(path, exist_ok=True)
    return os.path.join(os.getcwd(), path)
```
Creates a directory for storing cache files, with permissions to create intermediate directories if necessary.

### `return_df` <a name="return_df"></a>
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

### `find_all_summaries` <a name="find-all-summaries-detailed"></a>
```python
def find_all_summaries(model='all-mpnet-base-v2'):
    
    embedding_model = SentenceTransformer('all-mpnet-base-v2', cache_folder=model_custom_path)
    
    for i in range(len(list(project_details_df.project_id))):
        
        this_project_id = project_details_df.project_id[i]
        if this_project_id != this_project_id: # i.e., project_id is NaN
            this_project_id = project_details_df.generated_grant_id[i] # use the automatically generated grant ID instead
            
        # Find all rows where the corresponding 'project_id' column values are equal to this_project_id
        this_relevant_df = project_details_df.loc[project_details_df.project_id == this_project_id]
        this_project_context = ""
        this_project_context_embedding = None
        
        if this_relevant_df.empty:
            # Whatever we have right now is the only occurrence
            this_title = project_details_df.title[i]
            this_summary = project_details_df.summary[i]
        
            if not this_title != this_title and not this_summary != this_summary: # i.e., both title and summary are not NaN

                this_title_summary = this_title + '. ' + this_summary + ' ' # concatenate title and summary into 1 string
                # create embeddings with the concatenated string
                this_title_summary_embedding = embedding_model.encode(this_title_summary, convert_to_tensor=True)

                this_project_context = this_title_summary
                this_project_context_embedding = this_title_summary_embedding
            
        else:
            # Determine the necessary context required to capture information about this project by examining all existing titles and summaries
            this_project_context, this_project_context_embedding = generate_context_embeddings(this_relevant_df, embedding_model)
        
        store_context_and_embeddings(
            this_project_context, 
            this_project_context_embedding,
            bucket = EMBEDDINGS_BUCKET,
            data_key = this_project_id,
            embedding_model=embedding_model
        )
```
The `find_all_summaries` function is a core component of the script that processes project summaries to generate and store their embeddings. This function operates without direct inputs from function arguments, instead, it utilizes global variables predefined earlier in the script. Below is a detailed breakdown of its operations:

#### Process Flow
1. **Initialize Embedding Model**:
   - A SentenceTransformer model, specifically 'all-mpnet-base-v2', is loaded with a specified cache directory to save and retrieve model components efficiently. This model is used to generate embeddings from textual data (project summaries).

2. **Iterate Over Project Details**:
   - The function iterates through the DataFrame `project_details_df` that contains project details, including project IDs and summaries. This DataFrame is expected to be loaded beforehand by the `return_df` function, using parameters from AWS Glue.

3. **Process Each Project**:
   - For each project entry in the DataFrame:
     - The function checks if the project ID is missing (`NaN`). If so, it uses a generated grant ID as a fallback identifier.
     - It then locates all DataFrame entries that correspond to this project ID and aggregates them if necessary.

4. **Generate and Manage Context and Embeddings**:
   - If only one entry exists for a project (i.e., the DataFrame filtered on the project ID has only one row), the function directly uses the title and summary from this row to create an embedding.
   - If multiple entries exist, it calls `generate_context_embeddings` to concatenate and embed the aggregated context from all entries. This helps in creating a more comprehensive representation of the project.
   - The context string (concatenated title and summary) and its resulting embedding are then either updated or newly stored in S3 by calling `store_context_and_embeddings`.

5. **Store Context and Embeddings**:
   - The `store_context_and_embeddings` function is invoked with the project's context, the corresponding embedding, the bucket name, and a data key that typically represents the project ID. This function either updates existing embeddings in S3 if they are below a certain similarity threshold or stores new embeddings if none exist.

#### Inputs and Outputs
- **Inputs**: The function does not take parameters directly; it operates on global variables and depends on the state of `project_details_df`.
- **Outputs**: There are no return values as the function stores/updates data in an S3 bucket.

### `check_and_update_embeddings` <a name="check-and-update-embeddings-detailed"></a>
```python
def check_and_update_embeddings(bucket, data_key, current_context, current_embeddings, embedding_model, threshold=0.96):
    """
    Check if embeddings exist, compare them, and update if similarity is above threshold.
    """
    s3_client = boto3.client('s3')
    context_key = f'{data_key}.pkl'
    embeddings_key = f'{data_key}_embeddings.pkl'

    try:
        # Check if the context and embeddings files exist
        context_object = s3_client.get_object(Bucket=bucket, Key=context_key)
        embeddings_object = s3_client.get_object(Bucket=bucket, Key=embeddings_key)
        
        # Load existing context and embeddings
        existing_context = pickle.loads(context_object['Body'].read())
        existing_embeddings = pickle.loads(embeddings_object['Body'].read())

        # Compare embeddings using cosine similarity
        similarity = util.pytorch_cos_sim(current_embeddings, existing_embeddings)

        if similarity < threshold:
            # Update context and embeddings if similarity is low
            updated_context = existing_context + " " + current_context
            updated_embeddings = embedding_model.encode(updated_context, convert_to_tensor=True) # Update embeddings by running the embedding_model on the new context that combined existing and current contexts
            updated_embeddings = pickle.dumps(updated_embeddings)
            # Save updated context and embeddings
            s3_client.put_object(Bucket=bucket, Key=context_key, Body=pickle.dumps(updated_context))
            print(f"Stored in S3: Data_key = {context_key}. Stored embeddings shape = {updated_embeddings.shape}")
            s3_client.put_object(Bucket=bucket, Key=embeddings_key, Body=updated_embeddings)
            return True
        else:
            print(f"Skipping the update embeddings stage for {context_key}.")
            return True
        
    except s3_client.exceptions.NoSuchKey:
        # If the files do not exist, proceed with the original saving process
        return False
```
The `check_and_update_embeddings` function is designed to manage and update project embeddings stored on AWS S3. It evaluates the necessity of updating existing embeddings based on their similarity to newly generated embeddings and handles the updating process if required.

#### Detailed Process Flow
1. **Setup AWS S3 Client**:
   - Initializes a `boto3` client for interacting with AWS S3, which is used throughout the function to retrieve and store data.

2. **Define Key Locations**:
   - Constructs S3 object keys for both the context and embeddings using the provided `data_key`. This allows the function to target specific files in S3.

3. **Attempt Retrieval of Existing Data**:
   - Attempts to retrieve the existing context and embeddings from S3 using the constructed keys. If either does not exist (`NoSuchKey` exception), the function skips the update process and indicates that no existing data was found.

4. **Deserialize Data**:
   - If the context and embeddings are successfully retrieved, they are deserialized using `pickle` from the binary format stored in S3. This converts them back into Python objects that can be manipulated in the script.

5. **Compare Embeddings**:
   - Utilizes the `util.pytorch_cos_sim` function from the `SentenceTransformer` library to compute the cosine similarity between the newly generated embeddings and the existing ones. This similarity measurement helps determine how closely related the new context is to the old one.

6. **Decision on Updating**:
   - If the similarity is below the specified threshold (e.g., 0.96), it indicates that the new context provides sufficiently different information, and thus the embeddings should be updated.
   - Constructs a new combined context by appending the new context to the existing one.
   - Generates new embeddings for this updated context using the SentenceTransformer model.

7. **Serialize and Store Updated Data**:
   - Serializes the updated context and embeddings using `pickle` and uploads them back to S3, replacing the older versions.
   - Provides logs or print statements indicating the status of the storage, including the shape of the stored embeddings, which is crucial for verifying the update.

#### Inputs and Outputs
- **Inputs**:
  - `bucket`: The name of the S3 bucket where the data is stored.
  - `data_key`: The base key used to locate the context and embeddings in S3.
  - `current_context`: The newly generated context string.
  - `current_embeddings`: The tensor representing the newly generated embeddings.
  - `embedding_model`: The SentenceTransformer model used to generate embeddings.
  - `threshold`: The cosine similarity threshold below which updates should be made.

- **Outputs**:
  - The function returns `True` if it proceeds with an update or skips it due to high similarity, and `False` if no existing data was found, indicating that new data should be stored instead.

### `store_context_and_embeddings` <a name="store-context-and-embeddings-detailed"></a>
```python
def store_context_and_embeddings(project_context, project_context_embedding, bucket, data_key, embedding_model):
    if project_context_embedding is not None:  # If it is not empty
        if not check_and_update_embeddings(
            bucket=bucket, data_key=data_key, current_context=project_context, 
            current_embeddings=project_context_embedding, embedding_model=embedding_model
        ):
            # If embeddings under this name (data_key) do not exist already (also, if they do, then they that situation has already been address with the check_and_update_embeddings function)
            s3_client = boto3.client('s3')
            context_bytes = pickle.dumps(project_context)
            embeddings_bytes = pickle.dumps(project_context_embedding)

            context_key = f'{data_key}.pkl'
            embeddings_key = f'{data_key}_embeddings.pkl'
            
            s3_client.put_object(Bucket=bucket, Key=context_key, Body=context_bytes)
            print(f"Stored in S3: Data_key = {context_key}. Stored embeddings shape = {project_context_embedding.shape}")
            s3_client.put_object(Bucket=bucket, Key=embeddings_key, Body=embeddings_bytes)
```
The `store_context_and_embeddings` function performs the tasks of managing the storage and updating of project embeddings in AWS S3. This function ensures that the project context and its corresponding embeddings are either newly stored or appropriately updated in the S3 bucket.

#### Detailed Process Flow
1. **Check for Existing Embeddings**:
   - The function begins by invoking `check_and_update_embeddings` to determine if existing embeddings should be updated. This function checks the existing embeddings stored in S3 against the newly generated embeddings to see if the similarity is below a specified threshold.

2. **Decision on Storage or Update**:
   - If `check_and_update_embeddings` returns `False`, it indicates that no existing embeddings were found for the given data key. In this case, the function proceeds to store the new context and embeddings.
   - If `check_and_update_embeddings` returns `True`, it means that the existing embeddings have been successfully updated, and no further action is needed.

3. **Serialize New Data**:
   - If the decision is made to store new data, the function serializes the project context and embeddings into a binary format using `pickle`. This serialization is necessary to store the data in S3, which requires a byte-stream format.

4. **Generate S3 Object Keys**:
   - Constructs specific keys for storing the serialized context and embeddings. Typically, these keys are based on the project data key with appropriate suffixes to distinguish context from embeddings.

5. **Upload to S3**:
   - Uses the `boto3` S3 client to upload the serialized context and embeddings to the specified bucket under their respective keys. This process involves creating S3 objects with the binary data.

6. **Logging**:
   - The function logs or prints statements indicating the storage of the data, including details like the data key and the shape of the embeddings tensor.

#### Inputs and Outputs
- **Inputs**:
  - `project_context`: The text representing the project context.
  - `project_context_embedding`: The tensor of embeddings generated from the project context.
  - `bucket`: The S3 bucket name where the data is to be stored.
  - `data_key`: The identifier used to generate unique keys for storing the context and embeddings.
  - `embedding_model`: The SentenceTransformer model, which might be required if additional embedding generation is needed during the update check.

- **Outputs**:
  - There are no explicit return values from this function. Its primary output is the side effect of data being stored or updated in S3.

### `generate_context_embeddings`  <a name="generate-context-embeddings-detailed"></a>
```python
def generate_context_embeddings(relevant_df, embedding_model):
    
    context = ""
    context_embedding = None
    
    # iterate through each rows of the "relevant" df
    for index, row in relevant_df.iterrows():
        if context_embedding is None:
            current_context_embedding = embedding_model.encode(context, convert_to_tensor=True)
        else:
            current_context_embedding = context_embedding
        
        this_title = row.title
        this_summary = row.summary
        
        if not this_title != this_title and not this_summary != this_summary: # i.e., both title and summary are not NaN
            
            this_title_summary = this_title + '. ' + this_summary + ' '
            this_title_summary_embedding = embedding_model.encode(this_title_summary, convert_to_tensor=True)
            
            if util.cos_sim(current_context_embedding, this_title_summary_embedding)[0] < 0.96: # The new title+summary has some differences
                context += this_title_summary
                context_embedding = embedding_model.encode(context, convert_to_tensor=True)
                
    return context, context_embedding
```
The `generate_context_embeddings` function creates a contextual representation of a project by combining the information from multiple entries, such as titles and summaries. It then generates a corresponding embedding for this combined context using a SentenceTransformer model. This function is particularly useful when multiple entries for a single project need to be aggregated to form a more comprehensive understanding.

#### Detailed Process Flow
1. **Initialize Context Variables**:
   - Starts by initializing an empty string for `context` and setting `context_embedding` to `None`. These will store the aggregated textual context and its corresponding embedding.

2. **Iterate Through DataFrame Rows**:
   - Iterates through each row of a provided DataFrame (`relevant_df`) which contains multiple entries related to a single project. This DataFrame is filtered prior to the call, containing only relevant entries for a specific project ID.

3. **Concatenate Context**:
   - For each row, extracts the title and summary and checks if they are not `NaN`. Concatenates each valid title and summary to the `context` string with formatting to ensure proper sentence separation.

4. **Generate Embedding for Updated Context**:
   - After each concatenation, checks if a `context_embedding` already exists. If it does, the function updates this embedding by encoding the updated context using the SentenceTransformer model.
   - If no embedding has been created yet (i.e., `context_embedding` is `None`), the function generates a new embedding from the current `context`.

5. **Optimize Embedding Updates**:
   - The embedding is updated only if the newly added title and summary significantly change the context. This is determined by calculating the cosine similarity between the current `context_embedding` and the embedding of the newly added title-summary. If the similarity is below a threshold (e.g., 0.96), the context is considered significantly different, prompting an embedding update.

6. **Return Combined Context and Embedding**:
   - Once all entries have been processed, returns the combined `context` and its corresponding `context_embedding`.

#### Inputs and Outputs
- **Inputs**:
  - `relevant_df`: A pandas DataFrame containing relevant entries (like titles and summaries) for a specific project.
  - `embedding_model`: A pre-loaded SentenceTransformer model used to generate embeddings.

- **Outputs**:
  - Returns a tuple containing:
    - `context`: A string that aggregates all relevant titles and summaries.
    - `context_embedding`: A tensor representing the embedding of the combined context.
