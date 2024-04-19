# Table of Contents <a name="table-of-contents"></a>
1. [Project Summary and Process Documentation](#project-summary-and-process-documentation)
   - [Script Overview](#script-overview)
     - [Import Libraries](#import-libraries)
     - [AWS Configuration and Parameters](#aws-configuration-and-parameters)
     - [Helper Functions](#helper-functions)
     - [Main Data Processing Functions](#main-data-processing-functions)
     - [Execution Flow](#execution-flow)
2. [Main Function (`tlef_raw_data_preprocessing`)](#main-function-tlef_raw_data_preprocessing)
   - [Data Retrieval and Preprocessing](#data-retrieval-and-preprocessing)
   - [Mapping and Cleaning Data](#mapping-and-cleaning-data)
   - [Final Outputs and S3 Interaction](#final-outputs-and-s3-interaction)
3. [Detailed Function Descriptions](#detailed-function-descriptions)
   - [`course_info_mapping`](#course-info-mapping-detailed)
   - [`year_mapping`](#year-mapping-detailed)
   - [`term_mapping`](#term-mapping-detailed)
   - [`ner_mapping`](#ner-mapping-detailed)
   - [`retrieve_member_info`](#retrieve-member-info-detailed)
   - [`generate_faculty_engagement_xlsx`](#generate-faculty-engagement-xlsx-detailed)
   - [`generate_project_details_xlsx`](#generate-project-details-xlsx-detailed)

# Project Summary and Process Documentation <a name="project-summary-and-process-documentation"></a>

This documentation outlines the processes and methodologies implemented in the Python script designed to preprocess and analyze project data within an educational context, particularly focusing on faculty engagement and project details for TLEF (Teaching and Learning Enhancement Fund) projects.

## Script Overview <a name="script-overview"></a>

### Import Libraries <a name="import-libraries"></a>
The script imports various libraries for data manipulation, AWS interactions, and processing:
- `boto3`: AWS SDK for Python, used for interacting with AWS services like S3.
- `pandas`, `numpy`: For data handling and manipulation.
- `re`, `datetime`, `collections`, `sys`: For regular expressions, date and time operations, data structures, and system-specific operations, respectively.
- `fuzzywuzzy`: For text matching which is crucial in data cleaning and entity recognition.

### AWS Configuration and Parameters <a name="aws-configuration-and-parameters"></a>
- Configuration for AWS Glue and S3 interactions, retrieving necessary parameters like bucket names and data paths.

### Helper Functions <a name="helper-functions"></a>
- Functions to handle data retrieval (`return_df`), transform values (`empty_to_blank`), and manipulate text data for mapping and cleaning.

### Main Data Processing Functions <a name="main-data-processing-functions"></a>
- Functions that implement the core logic for data mapping (`course_info_mapping`, `year_mapping`), entity recognition (`ner_mapping`), and generation of structured outputs (`generate_faculty_engagement_xlsx`, `generate_project_details_xlsx`).

### Execution Flow <a name="execution-flow"></a>
- The script is structured to process data in several stages, from initial data retrieval to final data cleaning and output generation, with specific functions designated for each step in the process.

## Main Function (`tlef_raw_data_preprocessing`) <a name="main-function-tlef_raw_data_preprocessing"></a>

### Data Retrieval and Preprocessing <a name="data-retrieval-and-preprocessing"></a>
- **Initial Data Retrieval**: Data is fetched from S3 using the `return_df` function based on file type and path.
- **Basic Data Cleaning**: Applying transformations to handle empty values and irrelevant data.

### Mapping and Cleaning Data <a name="mapping-and-cleaning-data"></a>
- **Data Mapping**: Implementing mappings for course information, academic years, and terms using regular expressions and specific cleaning logic.
- **Entity Recognition**: Using natural language processing to identify and map team members' names and roles to their respective faculty and department information.

### Final Outputs and S3 Interaction <a name="final-outputs-and-s3-interaction"></a>
- **Output Generation**: Creating structured Excel files for faculty engagement and project details.
- **S3 Storing**: Output files are stored back to S3, segmented by their respective categories (faculty engagement, project details).

## Detailed Function Descriptions <a name="detailed-function-descriptions"></a>

### `course_info_mapping` <a name="course-info-mapping-detailed"></a>
```python
def course_info_mapping(df, column_list):
    course_info_pattern = re.compile(r'([A-Z]+\d+)([A-Z/]*)')
    for column_name in column_list:
        new_column_values = []
        for value_idx in range(len(list(df[column_name]))):
            this_course_info = ''.join(str(df[column_name].iloc[value_idx]).upper().split())
            course_info = re.findall(course_info_pattern, this_course_info)
            if course_info:
                course_code = course_info[0][0]
                suffix = course_info[0][1].strip('/')
                if suffix:
                    if '/' in suffix:
                        all_course_codes = []
                        detailed_codes = suffix.split('/')
                        for d_code in detailed_codes:
                            this_course_code = course_code + d_code
                            all_course_codes.append(course_info_formatting(this_course_code))
                        new_column_values.append(all_course_codes)
                        continue
                    else:
                        course_code += suffix
                course_code = course_info_formatting(course_code)
                new_column_values.append([course_code])
            else:
                new_column_values.append('')
        df['cleaned_'+column_name] = new_column_values
```
The `course_info_mapping` function systematically processes course information in a dataset to standardize course codes and their formats. This function is crucial for ensuring consistent data across records where course identifiers might vary in formatting or detail level.

#### Detailed Process Flow
1. **Regular Expression Setup**:
   - Defines a regular expression pattern to match typical course codes (e.g., "CS101") and optional suffixes (e.g., "A/B/C"), which may indicate different sections or semesters of the course.

2. **Iterate Over Columns**:
   - The function iterates over each column specified in `column_list`, applying the mapping logic to each. These columns are expected to contain course-related information.

3. **Process Each Entry**:
   - For each entry in the current column, it strips whitespace, converts to uppercase for standardization, and then applies the regular expression to find matches.
   - If matches are found, it further processes them to separate the base course code from any suffixes.

4. **Handle Suffixes**:
   - If suffixes are present, checks for multiple codes separated by slashes and handles them individually by appending them to the base course code and formatting each.
   - Appends all processed course codes to a list which is then set as the new value for that entry in the DataFrame.

5. **Store Processed Data**:
   - Creates a new column in the DataFrame with a prefix 'cleaned_' added to the original column name, storing the newly formatted course codes.

6. **Logging and Debugging**:
   - The function can be extended to log detailed processing information or to include debugging statements to trace the transformation of specific entries, facilitating maintenance and updates.

#### Inputs and Outputs
- **Inputs**:
  - `df`: The pandas DataFrame containing the data to be processed.
  - `column_list`: A list of column names in the DataFrame that contain course information needing standardization.

- **Outputs**:
  - The function modifies the DataFrame `df` in-place, adding new columns for each processed column, containing the cleaned and standardized course information. No value is returned; the primary output is the modified DataFrame.

### `year_mapping` <a name="year-mapping-detailed"></a>
```python
def year_mapping(df, column_list, year_start=0, upto_future_year=5):
    today = datetime.date.today()
    this_year = int(str(today.year)[-2:]) + upto_future_year
    years = [str(j) if j>=10 else '0'+str(j) for j in range(year_start, this_year)]
    for column_name in column_list:
        new_column_values = []
        for value_idx in range(len(list(df[column_name]))):
            mentioned_years = []
            this_year_data = str(df[column_name].iloc[value_idx])
            if this_year_data == '20245/2025':  # Exception handling for specific known data errors
                new_column_values.append('')
                continue
            this_year_data = process_colon(this_year_data)
            this_year_data = remove_except_numbers_dash_and_onward_no_space(this_year_data)
            this_year_data = remove_numeric_after_slash(this_year_data)
            this_year_data, is_2020 = check_string_for_20(this_year_data)
            if is_2020:
                mentioned_years.append('2020')
            for year in years:
                if str(year) in this_year_data:
                    mentioned_years.append('20'+str(year))
                    if "onward" in this_year_data:
                        today = datetime.date.today()
                        this_year = int(str(today.year)[-2:])
                        for elapsed_year in range(int(year)+1, this_year):
                            mentioned_years.append('20'+str(elapsed_year))
            if mentioned_years:
                mentioned_years = list(set(mentioned_years))
                if '-' in this_year_data:
                    mentioned_years = fill_in_years(mentioned_years)
                mentioned_years.sort()
                new_column_values.append(mentioned_years)
            else:
                new_column_values.append('')
        df['cleaned_'+column_name] = new_column_values
```
The `year_mapping` function processes and standardizes academic year data within a DataFrame, ensuring that year references are consistent and comprehensive across dataset entries. This function handles potentially fragmented or inconsistent year formats and normalizes them into a standardized list format.

#### Detailed Process Flow
1. **Initialize Year Range**:
   - Computes the current year and establishes a range of years from a starting year to a specified number of years into the future. This range is used to identify and parse year data within the dataset entries.

2. **Iterate Over Columns**:
   - The function iterates through each specified column that contains year data, applying transformation and mapping logic to each entry.

3. **Process Each Entry**:
   - For each entry in the column, it:
     - Handles known data errors explicitly, such as correcting specific typo errors.
     - Processes the text to extract year data, removing unwanted characters and formats through a series of helper functions (`process_colon`, `remove_except_numbers_dash_and_onward_no_space`, `remove_numeric_after_slash`).
     - Checks for and processes mentions of the year 2020 specifically, due to its significance or prevalence in the dataset.

4. **Identify and List Years**:
   - Identifies all year references in the processed text using the initialized year range.
   - Checks for textual markers like "onward" to infer continuous ranges and adjusts the list of years accordingly.
   - If a range is specified (e.g., "2018-2020"), it fills in the intervening years.

5. **Store Processed Years**:
   - Converts the list of mentioned years into a set and back to a list to remove duplicates, then sorts it.
   - Appends this list to a new column in the DataFrame specifically for cleaned year data, prefixed with 'cleaned_'.

6. **Logging and Debugging**:
   - For further traceability and maintenance, logging or debug statements could be added to track the transformations and catch any anomalies in the year data processing.

#### Inputs and Outputs
- **Inputs**:
  - `df`: The pandas DataFrame containing the data to be processed.
  - `column_list`: A list of column names in the DataFrame that contain year information.
  - `year_start`: The starting year for generating the year range (default is 0).
  - `upto_future_year`: The number of years into the future to include in the year range beyond the current year (default is 5).

- **Outputs**:
  - The function modifies the DataFrame `df` in-place, adding new columns for each processed column containing the cleaned and standardized list of years. No return value is provided; the primary output is the modified DataFrame.

### `term_mapping` <a name="term-mapping-detailed"></a>
```python
def term_mapping(df, column_list):
    terms = ['sep', 'jan', 'may', 'jul']
    for column_name in column_list:
        for value_idx in range(len(list(df[column_name]))):
            mentioned_terms = []
            for term in terms:
                if term in str(df[column_name].iloc[value_idx]).lower():
                    mentioned_terms.append(term)
            if mentioned_terms:
                df[column_name].iloc[value_idx] = ','.join(mentioned_terms)
            else:
                df[column_name].iloc[value_idx] = ''
```
The `term_mapping` function is designed to identify and standardize academic term references within a DataFrame. It simplifies the representation of terms in the dataset by normalizing diverse term descriptions into a consistent format, facilitating easier analysis and comparison.

#### Detailed Process Flow
1. **Define Term Keywords**:
   - A list of term keywords (`terms`) is defined, representing typical academic terms such as "Sep", "Jan", "May", and "Jul". These keywords are used to search and identify term references within the dataset entries.

2. **Iterate Over Columns**:
   - The function iterates through each specified column that contains term information, applying the identification logic to each entry.

3. **Identify Terms in Entries**:
   - For each entry in a column:
     - The function initializes an empty list, `mentioned_terms`, to collect the terms identified within the entry.
     - It loops through each term in the predefined `terms` list, checking if the term is present in the entry (converted to lowercase for case insensitivity).
     - If a term is found, it is added to the `mentioned_terms` list.

4. **Update DataFrame Entries**:
   - After processing an entry:
     - If terms were identified (`mentioned_terms` is not empty), the entry in the DataFrame is updated to a comma-separated string of the terms, providing a clean and unified format.
     - If no terms are identified, the entry is set to an empty string, indicating the absence of term information.

5. **In-Place DataFrame Modification**:
   - The modifications (updating or clearing entries) are made directly in the DataFrame `df`, altering it in-place without the need for creating a new DataFrame. This method is efficient in terms of memory usage and execution time.

6. **Logging and Debugging**:
   - To enhance maintainability and facilitate troubleshooting, the function could be equipped with logging statements to log the process of term identification and any anomalies encountered. This is especially useful in large datasets or when updates to term definitions or formats are made.

#### Inputs and Outputs
- **Inputs**:
  - `df`: The pandas DataFrame containing the data to be processed.
  - `column_list`: A list of column names in the DataFrame that are expected to contain term information.

- **Outputs**:
  - There is no return value; the function modifies the DataFrame `df` directly, updating it with cleaned and standardized term data. The main output is the enhanced clarity and uniformity in the representation of academic terms within the dataset.

### `ner_mapping` <a name="ner-mapping-detailed"></a>
```python
def ner_mapping(df, column, bucket, data_key, faculty_code_dict):
    df_institution_data = return_df(bucket, data_key)
    full_name_list = list(df_institution_data['PREFERRED_FULL_NAME'])
    department_list = list(df_institution_data['PRIMARY_DEPARTMENT_AFFILIATION'])
    faculty_list = list(df_institution_data['PRIMARY_FACULTY_AFFILIATION'])
    rank_list = list(df_institution_data['PRIMARY_ACADEMIC_RANK'])
    emails_list = list(df_institution_data['EMAIL_ADDRESS'])
    campus_list = list(df_institution_data['PRIMARY_CAMPUS_LOCATION'])
    
    source_data = df[column]
    destination_data = []
    
    for this_free_text_idx in range(len(source_data)):
        this_free_text = source_data.iloc[this_free_text_idx]
        if isinstance(this_free_text, str) and this_free_text != '': # only accepting valid strings
            associated_entities = retrieve_member_info(this_free_text)
            cleaned_associated_entities = add_institution_info(
                associated_entities,
                full_name_list,
                department_list,
                faculty_list,
                rank_list,
                emails_list,
                campus_list,
                faculty_code_dict
            )
            destination_data.append(cleaned_associated_entities)
        else:
            destination_data.append(this_free_text)
    df['cleaned_Team Members'] = destination_data
    return df
```
The `ner_mapping` function is essential for enriching dataset entries with structured and detailed information about entities (team members) identified within text data. It uses natural language processing to extract names, roles, and other pertinent details from unstructured text and maps these details to institutional data.

#### Detailed Process Flow
1. **Data Retrieval**:
   - Retrieves institutional data from an S3 bucket using the `return_df` function. This data includes lists of full names, department affiliations, faculty affiliations, academic ranks, email addresses, and campus locations.

2. **Initialize Processing Variables**:
   - Prepares lists from the retrieved institutional data, which will be used to match and enrich the identified entities.

3. **Iterate Over Data Entries**:
   - Processes each entry in the specified column of the DataFrame, ensuring that only valid (non-empty) strings are considered for entity extraction.

4. **Entity Extraction**:
   - For each valid text entry, the `retrieve_member_info` function is called to extract names, emails, and other entities using predefined patterns or NLP models.

5. **Enrich Extracted Data**:
   - Calls `add_institution_info` to map the extracted entity data to the corresponding institutional data based on matches found in names, emails, and other attributes. This function uses fuzzy matching to enhance the accuracy of data association, particularly useful in cases where exact matches are not possible.

6. **Store Enriched Data**:
   - Appends the enriched entity information to a list, which is then used to update the original DataFrame with a new column titled 'cleaned_Team Members', containing structured and detailed information about team members.

7. **Update DataFrame**:
   - The updated information is stored back in the DataFrame, replacing or appending to the existing data to reflect the enriched details accurately.

8. **Return Modified DataFrame**:
   - Returns the modified DataFrame, now enhanced with detailed entity information that can be directly used for further analysis or reporting.

#### Inputs and Outputs
- **Inputs**:
  - `df`: The pandas DataFrame containing the data to be processed.
  - `column`: The name of the column in `df` that contains the text data from which entities will be extracted.
  - `bucket`: The name of the S3 bucket from where institutional data is retrieved.
  - `data_key`: The key within the S3 bucket that specifies the exact file or dataset to retrieve.
  - `faculty_code_dict`: A dictionary mapping faculty names to their respective codes, used to standardize faculty references in the output.

- **Outputs**:
  - The function returns the DataFrame `df` with an additional or updated column 'cleaned_Team Members', which includes enriched and structured data about entities identified and mapped from the text. This enhanced dataset facilitates deeper insights and more effective data utilization.

### `retrieve_member_info` <a name="retrieve-member-info-detailed"></a>
```python
def retrieve_member_info(input_text):
    detected_entities = comprehend.detect_entities(Text=input_text, LanguageCode='en')['Entities']
    mentioned_faculties = get_faculty(input_text)
    faculties_idx = list(mentioned_faculties.keys())
    
    names_dict = {}
    names = []
    remaining_names = []
    
    emails = []
    emails_idx = []
    remaining_emails = []
    
    for entity in detected_entities:
        if entity['Type'] == 'PERSON':
            if ' ' in entity['Text'] and not ('President' in entity['Text'] or 'Dean' in entity['Text']):
                names_dict[entity['BeginOffset']] = entity['Text']
            
        elif entity['Type'] == 'OTHER':
            if '@' in entity['Text']:
                emails.append(entity['Text'])
                emails_idx.append(entity['BeginOffset'])
    
    names = list(names_dict.values())
    remaining_names = names.copy()
    remaining_emails = emails.copy()
    
    associated_entities = {}
    
    if len(names) == len(emails):
        for i in range(len(names)):
            associated_entities[names[i]] = {}
            associated_entities[names[i]]['Team Member Email'] = emails[i]
            remaining_names = []
            remaining_emails = []
            
    elif len(emails) < len(names):
        for i in range(len(emails)):
            remove_name = ""
            remove_email = ""
            
            this_fuzz_seq = {}
            this_fuzz_seq[0] = "blank" 
            for name in names:
                this_fuzz_ratio = fuzz.ratio(emails[i].split('@')[0].lower(), name.lower())
                if this_fuzz_ratio > list(this_fuzz_seq.keys())[0]:
                    del this_fuzz_seq[list(this_fuzz_seq.keys())[0]]
                    this_fuzz_seq[int(fuzz.ratio(emails[i].split('@')[0].lower(), name.lower()))] = name
                    
            associated_entities[this_fuzz_seq[list(this_fuzz_seq.keys())[0]]] = {} 
            associated_entities[this_fuzz_seq[list(this_fuzz_seq.keys())[0]]]['Team Member Email'] = emails[i]
            remove_name = str(this_fuzz_seq[list(this_fuzz_seq.keys())[0]])
            remove_email = emails[i]
            
            if remove_name in remaining_names:
                remaining_names.remove(remove_name)
            
            if remove_email in remaining_emails:
                remaining_emails.remove(remove_email)

        if remaining_names:
            for r_name in remaining_names:
                associated_entities[r_name] = {}
                associated_entities[r_name]['Team Member Email'] = ""
                
    elif len(names) < len(emails):
        for i in range(len(names)):
            remove_name = ""
            remove_email = ""
            
            this_fuzz_seq = {}
            this_fuzz_seq[0] = "blank" 
            for email in emails:
                this_fuzz_ratio = fuzz.ratio(email.split('@')[0].lower(), names[i].lower())
                if this_fuzz_ratio > list(this_fuzz_seq.keys())[0]:
                    del this_fuzz_seq[list(this_fuzz_seq.keys())[0]]
                    this_fuzz_seq[
                        int(fuzz.ratio(email.split('@')[0].lower(), names[i].lower()))] = names[i]
                    
            associated_entities[this_fuzz_seq[list(this_fuzz_seq.keys())[0]]] = {}
            associated_entities[this_fuzz_seq[list(this_fuzz_seq.keys())[0]]]['Team Member Email'] = email
            remove_name = str(this_fuzz_seq[list(this_fuzz_seq.keys())[0]])
            remove_email = email
            
            if remove_name in remaining_names:
                remaining_names.remove(remove_name)
            
            if remove_email in remaining_emails:
                remaining_emails.remove(remove_email)
    
    # Finally, update the faculty info
    for n_idx in names_dict.keys():
        higher_f_idx = [f_idx for f_idx in faculties_idx if f_idx > n_idx]
        if higher_f_idx:
            this_key = names_dict[n_idx]
            associated_entities[this_key]['Team Member Faculty'] = mentioned_faculties[min(higher_f_idx)]
            
    return associated_entities
```
The `retrieve_member_info` function is instrumental in extracting and organizing entity information from text data, specifically targeting team member names, emails, and faculty affiliations within project descriptions or related text content. This function leverages both natural language processing via AWS Comprehend and fuzzy matching techniques to enhance data quality and relevance.

#### Detailed Process Flow
1. **Entity Detection**:
   - Utilizes the AWS Comprehend service to detect entities within the provided text, focusing on recognizing person names and email addresses as primary entities of interest.

2. **Faculty Mention Identification**:
   - Calls the `get_faculty` function to identify mentions of faculty within the text, which aids in associating detected person entities with specific faculties based on proximity.

3. **Initialize Collections**:
   - Prepares lists to hold names and emails extracted, alongside their positions in the text, to facilitate association and ensure that entities are matched correctly based on their appearances in the text.

4. **Fuzzy Matching for Association**:
   - Implements fuzzy matching logic to associate names with emails accurately. This step is crucial when direct one-to-one matches between names and emails are unclear, using similarity scores to find the best matches.
   - Adjusts remaining unassociated names or emails based on the outcomes of these associations.

5. **Faculty Association**:
   - Enhances the name-entity data with faculty information by matching the proximity of faculty mentions to the position of the name entity within the text.

6. **Return Structured Entity Data**:
   - Constructs and returns a dictionary containing the structured entity information, with keys as person names and values as dictionaries holding their respective email, faculty, and other relevant information.

#### Inputs and Outputs
- **Inputs**:
  - `input_text`: A string containing the unstructured text from which entity information is to be extracted.

- **Outputs**:
  - Returns a dictionary where each key is a detected person name and each value is another dictionary containing attributes such as the team member's email, faculty, and any additional identified details. This output facilitates the enrichment of datasets with detailed and structured relational data about project participants.

### `generate_faculty_engagement_xlsx` <a name="generate-faculty-engagement-xlsx-detailed"></a>
```python
def generate_faculty_engagement_xlsx(df):
    funding_year = []
    project_type = []
    application_title = []
    project_faculty = []
    team_member_name = []
    team_member_title = []
    team_member_stream = []
    campus = []
    team_member_faculty = []
    team_member_department = []
    team_member_email = []

    for proj_idx in range(len(df)):
        this_member_info = df['cleaned_Team Members'].iloc[proj_idx]
        if not isinstance(this_member_info, int) and this_member_info != '':  # Not an integer or a blank string
            for this_team_member_name in list(this_member_info.keys()):
                funding_year.append(df['Funding Year'].iloc[proj_idx])
                project_type.append(df['Project Type'].iloc[proj_idx])
                application_title.append(df['Application Title'].iloc[proj_idx])
                project_faculty.append(df['Project Faculty'].iloc[proj_idx])

                team_member_name.append(this_team_member_name)

                if 'Team Member Title' in list(this_member_info[this_team_member_name].keys()):
                    team_member_title.append(this_member_info[this_team_member_name]['Team Member Title'])
                else:
                    team_member_title.append('')

                if 'Team Member Stream' in list(this_member_info[this_team_member_name].keys()):
                    team_member_stream.append(this_member_info[this_team_member_name]['Team Member Stream'])
                else:
                    team_member_stream.append('')

                if 'Campus' in list(this_member_info[this_team_member_name].keys()):
                    campus.append(this_member_info[this_team_member_name]['Campus'])
                else:
                    campus.append('')

                if 'Team Member Faculty' in list(this_member_info[this_team_member_name].keys()):
                    team_member_faculty.append(this_member_info[this_team_member_name]['Team Member Faculty'])
                else:
                    team_member_faculty.append('')

                if 'Team Member Department' in list(this_member_info[this_team_member_name].keys()):
                    team_member_department.append(this_member_info[this_team_member_name]['Team Member Department'])
                else:
                    team_member_department.append('')

                if 'Team Member Email' in list(this_member_info[this_team_member_name].keys()):
                    team_member_email.append(this_member_info[this_team_member_name]['Team Member Email'])
                else:
                    team_member_email.append('')

    n_rows = len(funding_year)

    faculty_engagement_df = pd.DataFrame({
        'funding_year': funding_year,
        'project_type': project_type,
        'project_id': application_title,
        'grant_id': [''] * n_rows,
        'project_faculty': project_faculty,
        'member_name': team_member_name,
        'member_title': team_member_title,
        'member_stream': team_member_stream,
        'member_campus': campus,
        'member_faculty': team_member_faculty,
        'member_unit': team_member_department,
        'member_other': [''] * n_rows
    })

    return faculty_engagement_df
```
The `generate_faculty_engagement_xlsx` function creates a structured Excel format dataset that details the engagement of faculty members in various projects. This function is essential for synthesizing comprehensive faculty data from enriched entity information extracted and processed from project data.

#### Detailed Process Flow
1. **Initialize Data Lists**:
   - Prepares lists to collect data on funding years, project types, application titles, and details about team members, including names, titles, departments, faculties, campuses, and email addresses.

2. **Iterate Over Projects**:
   - Loops through each project in the DataFrame, retrieving enriched team member data from the 'cleaned_Team Members' column.

3. **Extract Member Details**:
   - For each project, iterates over the keys (team member names) in the member information dictionary:
     - Extracts and appends detailed information about each team member, such as their title, faculty, department, campus, and contact email.

4. **Compile Data into DataFrame**:
   - Combines all collected data into a new pandas DataFrame, organizing it into columns that reflect each aspect of faculty engagement (e.g., funding year, project type, team member name, member title).

5. **Return Structured DataFrame**:
   - Returns the newly created DataFrame, which is ready to be exported to Excel or further analyzed. This structured format makes it easy to perform detailed analyses on faculty engagement across various projects.

#### Inputs and Outputs
- **Inputs**:
  - `df`: The pandas DataFrame containing processed and enriched data about projects and their team members.

- **Outputs**:
  - Returns a pandas DataFrame designed specifically for analyzing faculty engagement, with comprehensive details about each team member’s role and involvement in projects. This DataFrame can easily be saved to an Excel file for reporting or presentation purposes.

### `generate_project_details_xlsx` <a name="generate-project-details-xlsx-detailed"></a>
```python
def generate_project_details_xlsx(df):
    funding_year = df['Funding Year']
    n_rows = len(funding_year)
    project_type = df['Project Type']
    grant_id = df['Application Title']
    project_id = [''] * n_rows  # This info is not present in the raw survey monkey dataset
    project_faculty = df['Project Faculty']
    pi_name = df['PI']
    pi_unit = df['Department']
    funding_amount = df['Total Project Budget']
    title = df['Project Title']
    summary = df['Summary of Work Accomplished to Date (1000 words max.)']
    co_applicants = df['Team Members']
    generated_grant_id = [''] * n_rows  # This info is not generated at this stage
    project_year = [''] * n_rows  # This info is not present in the raw survey monkey dataset
    project_status = [''] * n_rows  # This info is not present in the raw survey monkey dataset

    project_details_df = pd.DataFrame({
        'funding_year': funding_year,
        'project_type': project_type,
        'grant_id': grant_id,
        'project_id': project_id,
        'project_faculty': project_faculty,
        'pi_name': pi_name,
        'pi_unit': pi_unit,
        'funding_amount': funding_amount,
        'title': title,
        'summary': summary,
        'co_applicants': co_applicants,
        'generated_grant_id': generated_grant_id,
        'project_year': project_year,
        'project_status': project_status
    })

    return project_details_df
```
The `generate_project_details_xlsx` function creates a structured DataFrame that encapsulates comprehensive details about each project, including information about funding, project leads, and their respective departments. This function is crucial for providing an organized overview of project details which facilitates further analysis or reporting.

#### Detailed Process Flow
1. **Extract Data from Source DataFrame**:
   - Pulls relevant data directly from the source DataFrame `df` for various columns like funding year, project type, grant IDs, and more.

2. **Initialize Missing Data Columns**:
   - Sets up columns for which data may not be directly available in the source DataFrame (`project_id`, `generated_grant_id`, `project_year`, `project_status`). These are initialized as empty strings for all rows to maintain consistency in DataFrame structure.

3. **Compile Project Details into New DataFrame**:
   - Combines all the extracted and initialized data into a new pandas DataFrame. This DataFrame is specifically structured to capture all essential details about the projects, aligning each project's various attributes into a row-based format.

4. **Return Structured DataFrame**:
   - The function returns this newly created DataFrame, which is well-suited for exporting to an Excel format for distribution or detailed analysis.

#### Inputs and Outputs
- **Inputs**:
  - `df`: The pandas DataFrame containing processed and cleaned data about projects. This DataFrame is expected to have columns that align with the various data elements the function extracts.

- **Outputs**:
  - Returns a pandas DataFrame that includes detailed and structured information about each project. This DataFrame is particularly useful for stakeholders who need a comprehensive overview of project specifics, including funding, principal investigators, and project outcomes. The format is ready for direct export to Excel or integration into reports and presentations.

[🔼 Back to top](#table-of-contents)
