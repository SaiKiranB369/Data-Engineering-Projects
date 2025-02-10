import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import lit
import logging
import configparser
from datetime import datetime
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType
import shutil
import re

# Initialize Spark session
spark = SparkSession.builder.appName("SilverLayerProcessing").getOrCreate()

def read_config():
    """Load config2.ini from the project root directory."""
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Get script directory
    config_path = os.path.join(script_dir, "config2.ini")

    if not os.path.exists(config_path):  # Check if file exists
        raise FileNotFoundError(f"Error: config2.ini not found at {config_path}")

    config_obj = configparser.ConfigParser()
    config_obj.read(config_path)  # Read config file

    if not config_obj.sections():  # Ensure config isn't empty
        raise ValueError("Error: config2.ini is empty or misformatted!")

    return config_obj

# Load config2.ini
config = read_config()

# Retrieve paths
silver_input = config['PATHS']['input_path_silver_processing_layer']
silver_output = config['PATHS']['output_path_silver_processing_layer']
log_file = config['PATHS']['logs_folder']

# Configure logging
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("Logging initialized....")

# Define schema for Silver Layer
customer_schema = StructType([
    StructField("customer_id", IntegerType(), True),
    StructField("full_name", StringType(), True),
    StructField("email", StringType(), True),
    StructField("address", StringType(), True),
    StructField("city", StringType(), True),
    StructField("state", StringType(), True),
    StructField("ingest_timestamp", StringType(), True),
    StructField("source_system", StringType(), True),
])

product_schema = StructType([
    StructField("product_id", IntegerType(), True),
    StructField("product_name", StringType(), True),
    StructField("category", StringType(), True),
    StructField("price", FloatType(), True),
    StructField("ingest_timestamp", StringType(), True),
    StructField("source_system", StringType(), True),
])

order_schema = StructType([
    StructField("order_id", IntegerType(), True),
    StructField("order_date", StringType(), True),
    StructField("customer_id", IntegerType(), True),
    StructField("total_amount", FloatType(), True),
    StructField("order_status", StringType(), True),
    StructField("ingest_timestamp", StringType(), True),
    StructField("source_system", StringType(), True),
])

order_item_schema = StructType([
    StructField("order_item_id", IntegerType(), True),
    StructField("order_id", IntegerType(), True),
    StructField("product_id", IntegerType(), True),
    StructField("quantity", IntegerType(), True),
    StructField("unit_price", FloatType(), True),
    StructField("ingest_timestamp", StringType(), True),
    StructField("source_system", StringType(), True),
])

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

def get_processed_timestamps(silver_output):
    """Retrieve timestamps of already processed files."""
    processed_timestamps = set()
    for root, dirs, files in os.walk(silver_output):
        for file in files:
            if file.endswith('.csv'):
                timestamp = extract_timestamp_from_filename(file)
                if timestamp:
                    processed_timestamps.add(timestamp)
    return processed_timestamps

def process_silver_layer():
    """Processes raw CSVs into Silver layer with cleaning and referential integrity enforcement."""
    # Extract all subfolders (dates) from bronze input folders
    bronze_files = {
        "customers": os.path.join(silver_input, "bronze_customers"),
        "orders": os.path.join(silver_input, "bronze_orders"),
        "order_items": os.path.join(silver_input, "bronze_order_items"),
        "products": os.path.join(silver_input, "bronze_products")
    }

    # Get all the available date folders
    bronze_dates = set()
    for folder in bronze_files.values():
        for subfolder in os.listdir(folder):
            if os.path.isdir(os.path.join(folder, subfolder)):  # Check for valid date subfolders
                bronze_dates.add(subfolder)

    # Initialize paths for silver and quarantine folders
    silver_customers_folder = os.path.join(silver_output, "silver_customers")
    silver_orders_folder = os.path.join(silver_output, "silver_orders")
    silver_order_items_folder = os.path.join(silver_output, "silver_order_items")
    silver_products_folder = os.path.join(silver_output, "silver_products")

    # Create directories if they do not exist
    os.makedirs(silver_customers_folder, exist_ok=True)
    os.makedirs(silver_orders_folder, exist_ok=True)
    os.makedirs(silver_order_items_folder, exist_ok=True)
    os.makedirs(silver_products_folder, exist_ok=True)

    # Get timestamps of already processed files
    processed_timestamps = get_processed_timestamps(silver_output)

    # Loop through each date and process data
    for bronze_date in bronze_dates:
        logging.info(f"Processing data for date: {bronze_date}")

        # Initialize quarantine folders
        quarantine_orders_folder = os.path.join(silver_orders_folder, bronze_date, "quarantine_orders")
        quarantine_order_items_folder = os.path.join(silver_order_items_folder, bronze_date, "quarantine_order_items")

        # Create directories for the date-specific folders
        os.makedirs(quarantine_orders_folder, exist_ok=True)
        os.makedirs(quarantine_order_items_folder, exist_ok=True)

        # Load data from bronze files for the specific date
        bronze_data = {}
        for file, folder in bronze_files.items():
            file_path = os.path.join(folder, bronze_date)
            all_files = [os.path.join(file_path, f) for f in os.listdir(file_path) if f.endswith('.csv')]

            if all_files:
                # Read and combine all files for the dataset
                df_list = []
                timestamps = []
                for file_path in all_files:
                    ingest_timestamp = extract_timestamp_from_filename(os.path.basename(file_path))
                    if ingest_timestamp in processed_timestamps:
                        logging.info(f"Skipping already processed file: {file_path}")
                        continue
                    df = spark.read.csv(file_path, header=True, inferSchema=True)
                    # Add timestamp to the dataframe
                    df = df.withColumn("ingest_timestamp", lit(ingest_timestamp))
                    df_list.append(df)
                    timestamps.append(ingest_timestamp)

                if df_list:
                    # Combine all DataFrames for the dataset
                    bronze_data[file] = df_list[0] if len(df_list) == 1 else df_list[0].unionByName(*df_list[1:])
                    # Store the earliest timestamp for the dataset
                    bronze_data[f"{file}_timestamp"] = min(timestamps)
                else:
                    logging.warning(f"No new files to process in {file_path}")
                    bronze_data[file] = spark.createDataFrame([], customer_schema)  # Empty dataframe to prevent errors
            else:
                logging.warning(f"No files found in {file_path}")
                bronze_data[file] = spark.createDataFrame([], customer_schema)  # Empty dataframe to prevent errors

        # Referential Integrity Checks for Orders and Order Items
        quarantined_orders = spark.createDataFrame([], order_schema)
        quarantined_order_items = spark.createDataFrame([], order_item_schema)

        if bronze_data["customers"].count() > 0 and bronze_data["orders"].count() > 0:
            valid_orders = bronze_data["orders"].join(bronze_data["customers"], "customer_id", "inner")
            logging.info(f"Valid orders count after join: {valid_orders.count()}")
            print(f"Valid orders count after join: {valid_orders.count()}")

            valid_orders = valid_orders.drop(bronze_data["customers"].ingest_timestamp)
            valid_orders = valid_orders.drop(bronze_data["customers"].source_system)
            valid_orders = valid_orders.withColumn("ingest_timestamp", lit(bronze_data["orders_timestamp"]))
            valid_orders = valid_orders.select(*bronze_data["orders"].columns)

            quarantined_orders = bronze_data["orders"].join(valid_orders, "order_id", "left_anti")
            logging.info(f"Invalid (quarantined) orders count: {quarantined_orders.count()}")
            print(f"Invalid (quarantined) orders count: {quarantined_orders.count()}")

            bronze_data["orders"] = valid_orders

        if bronze_data["orders"].count() > 0 and bronze_data["order_items"].count() > 0:
            valid_order_items = bronze_data["order_items"].join(bronze_data["orders"], "order_id", "inner")
            logging.info(f"Valid order items count after join: {valid_order_items.count()}")
            print(f"Valid order items count after join: {valid_order_items.count()}")

            valid_order_items = valid_order_items.drop(bronze_data["orders"].ingest_timestamp)
            valid_order_items = valid_order_items.drop(bronze_data["orders"].source_system)
            valid_order_items = valid_order_items.withColumn("ingest_timestamp", lit(bronze_data["order_items_timestamp"]))
            valid_order_items = valid_order_items.select(*bronze_data["order_items"].columns)

            quarantined_order_items = bronze_data["order_items"].join(valid_order_items, "order_item_id", "left_anti")
            logging.info(f"Invalid (quarantined) orders count: {quarantined_order_items.count()}")
            print(f"Invalid (quarantined) order items count: {quarantined_order_items.count()}")

            bronze_data["order_items"] = valid_order_items

        # Save valid data to silver layer
        # Save valid data to silver layer
        if any([df.count() > 0 for key, df in bronze_data.items() if not key.endswith("_timestamp")]):
            for file, df in bronze_data.items():
                if file.endswith("_timestamp"):  # Skip timestamp entries
                    continue
                if df.count() > 0:
                    # Extract the time part from the timestamp (HHMMSS)
                    timestamp = bronze_data[f"{file}_timestamp"].split("T")[1].replace(":", "")
                    silver_temp_path = os.path.join(silver_output, f"silver_{file}_temp_{timestamp}")
                    final_silver_path = os.path.join(silver_output, f"silver_{file}", bronze_date,
                                                     f"silver_{file}_{bronze_date}_{timestamp}.csv")
                    # Ensure the final destination directory exists before moving the file
                    os.makedirs(os.path.dirname(final_silver_path), exist_ok=True)
                    df.coalesce(1).write.csv(silver_temp_path, header=True, mode="overwrite")

                    # Move the part file to the final destination
                    for filename in os.listdir(silver_temp_path):
                        if filename.startswith("part-"):
                            shutil.move(os.path.join(silver_temp_path, filename), final_silver_path)
                            break

                    # Clean up the temp directory
                    shutil.rmtree(silver_temp_path)

                    logging.info(f"Saved silver file: {final_silver_path}")

            # Save quarantined data if any exists
            if quarantined_orders.count() > 0:
                # Extract the time part from the timestamp (HHMMSS)
                timestamp = bronze_data["orders_timestamp"].split("T")[1].replace(":", "")
                quarantine_orders_temp_path = os.path.join(quarantine_orders_folder,
                                                           f"quarantine_orders_temp_{timestamp}")
                final_quarantine_orders_path = os.path.join(quarantine_orders_folder,
                                                            f"quarantine_orders_{bronze_date}_{timestamp}.csv")
                quarantined_orders.coalesce(1).write.csv(quarantine_orders_temp_path, header=True, mode="overwrite")

                # Move the part file to the final destination
                for filename in os.listdir(quarantine_orders_temp_path):
                    if filename.startswith("part-"):
                        shutil.move(os.path.join(quarantine_orders_temp_path, filename), final_quarantine_orders_path)
                        break

                # Clean up the temp directory
                shutil.rmtree(quarantine_orders_temp_path)
                logging.info(f"Saved quarantine file: {final_quarantine_orders_path}")

            if quarantined_order_items.count() > 0:
                # Extract the time part from the timestamp (HHMMSS)
                timestamp = bronze_data["order_items_timestamp"].split("T")[1].replace(":", "")
                quarantine_order_items_temp_path = os.path.join(quarantine_order_items_folder,
                                                                f"quarantine_order_items_temp_{timestamp}")
                final_quarantine_order_items_path = os.path.join(quarantine_order_items_folder,
                                                                 f"quarantine_order_items_{bronze_date}_{timestamp}.csv")
                quarantined_order_items.coalesce(1).write.csv(quarantine_order_items_temp_path, header=True,
                                                              mode="overwrite")

                # Move the part file to the final destination
                for filename in os.listdir(quarantine_order_items_temp_path):
                    if filename.startswith("part-"):
                        shutil.move(os.path.join(quarantine_order_items_temp_path, filename),
                                    final_quarantine_order_items_path)
                        break

                # Clean up the temp directory
                shutil.rmtree(quarantine_order_items_temp_path)
                logging.info(f"Saved quarantine file: {final_quarantine_order_items_path}")
        else:
            logging.warning(f"No valid data found for silver layer processing for date: {bronze_date}")


if __name__ == "__main__":
    process_silver_layer()