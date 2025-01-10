## Modified By: Callam Josef Jackson-Sem
## Project: Lotto Predictor
## Purpose of File: To Process Historical Lottery Draw Data
## Description:
## This file handles the first step of the pipeline: processing historical lottery draw data.
## It retrieves, validates, and filters the past lottery results provided by the user or database.
## The processed data is stored in the pipeline for use by subsequent steps, ensuring that only valid draws
## are passed forward.

def process_historical_data(results, pipeline):
    """
    Processes historical lottery draw data and integrates it into the data pipeline.
    
    This function extracts past lottery results from the provided input, filters out invalid draws 
    (e.g., draws with invalid Powerball numbers), and stores the clean data in the pipeline for 
    downstream use.

    Parameters:
    - results (dict): A dictionary containing past lottery draw data under the key "past_results". 
                      Each draw is expected to be a dictionary with a "powerball" key, and optionally
                      other keys like "numbers", "bonus", etc.
    - pipeline (DataPipeline): The data pipeline object that facilitates data sharing between different 
                               stages of the pipeline.

    Returns:
    - None: The function directly modifies the pipeline, adding the cleaned data under the key "historical_data".
            If no valid historical data is found, it stores an empty list under this key.
    """

    # Step 1: Retrieves the list of past lottery draws from the input `results`.
    # If the key "past_results" does not exist in the dictionary, default to an empty list.
    historical_data = results.get("past_results", [])
    
    # Step 2: Check if any historical data has been provided.
    if not historical_data:
        ## Case: No historical data available
        # If the input is empty, inform the user and store an empty list in the pipeline.
        # Note: Using print here for simplicity; in production, replace this with a proper logging system.
        print("No historical data provided to pipeline.")
        
        # Add an empty list to the pipeline to indicate the absence of historical data.
        pipeline.add_data("historical_data", [])
        return  # Exit the function early since there is no data to process.

    # Step 3: Validates the historical data.
    ## Valid Powerball Range: Powerball numbers must be integers between 1 and 10.
    # Uses a list comprehension to filter out invalid draws.
    valid_historical_data = [
        draw for draw in historical_data
        if 1 <= draw.get("powerball", 0) <= 10  # Use 0 as a fallback if the "powerball" key is missing.
    ]

    # Step 4: Stores the filtered data into the pipeline.
    ## Adds the validated data to the pipeline under the key "historical_data".
    pipeline.add_data("historical_data", valid_historical_data)

    # Informs the user or logs that the processing step is complete.
    if valid_historical_data:
        print(f"Processed {len(valid_historical_data)} valid historical draws into the pipeline.")
    else:
        print("No valid historical draws found after filtering. Stored an empty list in the pipeline.")