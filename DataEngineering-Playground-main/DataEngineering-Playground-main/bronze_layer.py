import os
import logging
import shutil
from pyspark.sql import SparkSession
from pyspark.sql.functions import lit
import configparser
from datetime import datetime
import re

# Initialize Spark Session
spark = SparkSession.builder.appName("BronzeLayerProcessing").getOrCreate()


def read_config():
    """Load config2.ini from the project root directory."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "config2.ini")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Error: config2.ini not found at {config_path}")

    config_obj = configparser.ConfigParser()
    config_obj.read(config_path)

    if not config_obj.sections():  # Ensure config isn't empty
        raise ValueError("Error: config.ini is empty or misformatted!")

    return config_obj

# Read the config
config = read_config()

# paths from config2.ini
bronze_input = config['PATHS']['input_path_bronze_processing_layer']
bronze_output = config['PATHS']['output_path_bronze_processing_layer']
log_file = config['PATHS']['logs_folder']

def check_and_create_folders(config):
    """Check and create the necessary folders for input, output, and logs."""

    # Get paths from config
    bronze_output = config['PATHS']['output_path_bronze_processing_layer']
    log_file = config['PATHS']['logs_folder']

    if not os.path.exists(bronze_output):
        os.makedirs(bronze_output, exist_ok=True)  # Creates the directory if it doesn't exist
        logging.info(f"Created bronze output folder: {bronze_output}")
        print(f"Created bronze output folder: {bronze_output}")


# Ensure the logs folder exists (if the log_file contains a full path)
log_dir = os.path.dirname(log_file)
if not os.path.exists(log_dir):
    os.makedirs(log_dir, exist_ok=True)  # Creates the directory if it doesn't exist
    logging.info(f"Created logs folder: {log_dir}")
    print(f"Created logs folder: {log_dir}")

# Configure logging to use the exact log file path
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("Logging initialized....")
print("Logging initialized....")

# Check and create folders
check_and_create_folders(config)


def get_unprocessed_folders(raw_base_path, bronze_base_path):
    """Finds raw folders that haven't been processed in Bronze."""

    if not os.path.exists(raw_base_path) or not os.listdir(raw_base_path):
        logging.info("Raw Data generation not yet started")
        return "Raw Data generation not yet started"
        print("Raw Data generation not yet started")


    unprocessed_folders = []

    for dataset_folder in os.listdir(raw_base_path):
        raw_dataset_folder = os.path.join(raw_base_path, dataset_folder)

        if os.path.isdir(raw_dataset_folder):
            logging.info(f"Processing dataset: {dataset_folder}")

            for date_folder in os.listdir(raw_dataset_folder):
                raw_date_path = os.path.join(raw_dataset_folder, date_folder)
                bronze_date_path = os.path.join(bronze_base_path, "bronze_" + dataset_folder.replace("raw_", ""),
                                                date_folder)

                if os.path.isdir(raw_date_path):
                    for raw_file in os.listdir(raw_date_path):
                        if raw_file.endswith(".csv"):
                            bronze_file_path = os.path.join(bronze_date_path, raw_file.replace("raw_", "bronze_"))

                            if not os.path.exists(bronze_file_path):
                                unprocessed_folders.append(
                                    (date_folder, raw_file, raw_date_path, bronze_date_path, bronze_file_path))

    if not unprocessed_folders:
        logging.info("No new raw data available to process")
        print("No new raw data available to process")
        return []

    logging.info(f"Found {len(unprocessed_folders)} unprocessed files.")
    print(f"Found {len(unprocessed_folders)} unprocessed files.")
    return unprocessed_folders


def extract_timestamp_from_filename(filename):
    """
    Extracts the timestamp from a filename following the pattern:
    <dataset>_YYYYMMDD_HHMMSS.csv
    Example: customers_20240205_153045.csv → 2024-02-05T15:30:45
    """
    match = re.search(r'_(\d{8})_(\d{6})', filename)
    if match:
        date_part = match.group(1)  # YYYYMMDD
        time_part = match.group(2)  # HHMMSS
        ingest_timestamp = datetime.strptime(f"{date_part} {time_part}", "%Y%m%d %H%M%S").isoformat()
        return ingest_timestamp
    return None


def process_bronze_layer(raw_base_path, bronze_base_path):
    """Processes only unprocessed raw CSVs into the Bronze layer."""

    unprocessed_files = get_unprocessed_folders(raw_base_path, bronze_base_path)

    if not unprocessed_files:
        logging.info("No new raw files to process.")
        return

    # Process each file individually based on its own schema
    for date_folder, raw_file, raw_date_path, bronze_date_path, bronze_file_path in unprocessed_files:
        os.makedirs(bronze_date_path, exist_ok=True)

        # Read the raw CSV into a DataFrame with inferred schema
        try:
            df = spark.read.csv(os.path.join(raw_date_path, raw_file), header=True, inferSchema=True)
        except Exception as e:
            logging.error(f"Error reading file {raw_file}: {e}")
            continue

        # Extract timestamp from the file name
        ingest_timestamp = extract_timestamp_from_filename(raw_file)
        if not ingest_timestamp:
            logging.warning(f"Could not extract timestamp from {raw_file}, using current timestamp instead.")
            ingest_timestamp = datetime.now().isoformat()

        # Add static columns like ingest_timestamp and source_system
        df = df.withColumn("ingest_timestamp", lit(ingest_timestamp))
        df = df.withColumn("source_system", lit("DataIngestionSystem"))


        # Get the schema of the current file
        current_schema = set(df.columns)

        # Log schema info
        logging.info(f"Processing file {raw_file} with schema: {current_schema}")

        # Handle columns dynamically: Keep the columns from the file's schema only
        # Create a DataFrame with only the columns that exist in the file
        columns_to_keep = list(current_schema)

        # Write to temporary path first
        temp_path = os.path.join(bronze_date_path, f"temp_{raw_file}")
        df.coalesce(1).select(*columns_to_keep).write.csv(temp_path, header=True, mode="overwrite")

        # Move the processed file to the bronze folder
        for filename in os.listdir(temp_path):
            if filename.startswith("part-"):
                shutil.move(os.path.join(temp_path, filename), bronze_file_path)
                break

        # Clean up the temporary path
        shutil.rmtree(temp_path)
        logging.info(f"Processed: {bronze_file_path}")
        print(f"Processed: {bronze_file_path}")
    logging.info(f"All the unprocessed raw files are processed and saved in {bronze_output}")
    print(f"All the unprocessed raw files are processed and saved in {bronze_output}")


if __name__ == "__main__":
    process_bronze_layer(bronze_input, bronze_output)