import os
import shutil
import configparser
import logging

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType

# Initialize Spark session
spark = SparkSession.builder \
    .appName("GoldLayerProcessing") \
    .getOrCreate()


def read_config():
    """Load config.ini from the project root directory."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "config2.ini")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Error: config.ini not found at {config_path}")
    config_obj = configparser.ConfigParser()
    config_obj.read(config_path)
    if not config_obj.sections():
        raise ValueError("Error: config.ini is empty or misformatted!")
    return config_obj


# Load config.ini
config = read_config()

# Retrieve paths from config
gold_input = config['PATHS']['input_path_gold_processing_layer']
gold_output = config['PATHS']['output_path_gold_processing_layer']


log_file = config['PATHS']['logs_folder']
logging.basicConfig(filename=log_file, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def get_date_folders_for_entity(entity):
    """
    Given an entity name (e.g., 'silver_customers'), return a sorted list
    of date folders (ignoring any quarantine folders) in that entity folder.
    """
    entity_path = os.path.join(gold_input, entity)
    if not os.path.exists(entity_path):
        logging.warning(f"Entity folder {entity_path} does not exist.")
        return []
    # Only consider folders that do not start with "quarantine"
    folders = [d for d in os.listdir(entity_path)
               if os.path.isdir(os.path.join(entity_path, d)) and not d.startswith("quarantine")]
    return sorted(folders)


def find_silver_file(base_path, prefix, date, quarantine_subfolder=None):
    """
    Find a silver file in the given base_path with a file name that starts with prefix and date.
    If not found and quarantine_subfolder is provided, look inside the quarantine folder.
    """
    if os.path.exists(base_path):
        for file in os.listdir(base_path):
            full_path = os.path.join(base_path, file)
            if os.path.isfile(full_path) and file.startswith(f"{prefix}_{date}") and file.endswith(".csv"):
                return full_path
    # Check in the quarantine folder if provided
    if quarantine_subfolder:
        quarantine_path = os.path.join(base_path, quarantine_subfolder)
        if os.path.exists(quarantine_path):
            for file in os.listdir(quarantine_path):
                full_path = os.path.join(quarantine_path, file)
                if os.path.isfile(full_path) and file.startswith(f"quarantine_{prefix}_{date}") and file.endswith(
                        ".csv"):
                    return full_path
    return None


def extract_timestamp(file_path):
    """
    Extract the timestamp from the silver file name.
    Expected formats:
      - Primary file: "silver_<entity>_<date>_<time>.csv"
      - Quarantine file: "quarantine_<entity>_<date>_<time>.csv"
    Returns a string in the format "<date>_<time>" (e.g. "20250206_094453").
    """
    filename = os.path.basename(file_path)
    name_without_ext = filename[:-4] if filename.endswith(".csv") else filename
    parts = name_without_ext.split('_')
    if len(parts) >= 4:
        # parts[0] is either "silver" or "quarantine"
        # parts[1] is the entity name, parts[2] is the date and parts[3] is the time.
        return f"{parts[2]}_{parts[3]}"
    return None


def load_silver_data(date):
    """
    Load Silver layer data for the given date across all entities.
    It checks for the primary file and, if not found, for orders and order_items, looks in the quarantine folder.
    Also extracts the silver timestamp from one of the files (orders file is preferred).
    """
    # Define schemas for each table.
    # Note: We add ingest_timestamp as an optional field for orders so that if it is present in the silver file, it is preserved.
    schemas = {
        "customers": StructType([
            StructField("customer_id", StringType(), True),
            StructField("full_name", StringType(), True),
            StructField("email", StringType(), True),
            StructField("address", StringType(), True),
            StructField("city", StringType(), True),
            StructField("state", StringType(), True),
        ]),
        "products": StructType([
            StructField("product_id", StringType(), True),
            StructField("product_name", StringType(), True),
            StructField("category", StringType(), True),
            StructField("price", DoubleType(), True),
        ]),
        "orders": StructType([
            StructField("order_id", StringType(), True),
            StructField("order_date", StringType(), True),
            StructField("customer_id", StringType(), True),
            StructField("total_amount", DoubleType(), True),
            StructField("order_status", StringType(), True),
            # Presume silver orders already contain these columns, if available.
            StructField("source_system", StringType(), True),
            StructField("ingest_timestamp", StringType(), True)
        ]),
        "order_items": StructType([
            StructField("order_item_id", StringType(), True),
            StructField("order_id", StringType(), True),
            StructField("product_id", StringType(), True),
            StructField("quantity", IntegerType(), True),
            StructField("unit_price", DoubleType(), True),
        ]),
    }

    data = {}

    # Load customers
    cust_base = os.path.join(gold_input, "silver_customers", date)
    cust_file = find_silver_file(cust_base, "silver_customers", date)
    if cust_file:
        data["customers"] = spark.read.csv(cust_file, header=True, inferSchema=True)
        logging.info(f"Loaded customers from {cust_file}")
    else:
        logging.warning(f"Missing Silver customers file for date {date} in {cust_base}")
        data["customers"] = spark.createDataFrame([], schemas["customers"])

    # Load products
    prod_base = os.path.join(gold_input, "silver_products", date)
    prod_file = find_silver_file(prod_base, "silver_products", date)
    if prod_file:
        data["products"] = spark.read.csv(prod_file, header=True, inferSchema=True)
        logging.info(f"Loaded products from {prod_file}")
    else:
        logging.warning(f"Missing Silver products file for date {date} in {prod_base}")
        data["products"] = spark.createDataFrame([], schemas["products"])

    # Load orders (check quarantine if not found in main folder)
    orders_base = os.path.join(gold_input, "silver_orders", date)
    orders_file = find_silver_file(orders_base, "silver_orders", date, quarantine_subfolder="quarantine_orders")
    if orders_file:
        data["orders"] = spark.read.csv(orders_file, header=True, inferSchema=True)
        silver_timestamp = extract_timestamp(orders_file)
        data["silver_timestamp"] = silver_timestamp
        logging.info(f"Loaded orders from {orders_file} with silver timestamp: {silver_timestamp}")
    else:
        logging.warning(f"Missing Silver orders file for date {date} in {orders_base}")
        data["orders"] = spark.createDataFrame([], schemas["orders"])
        data["silver_timestamp"] = None

    # Load order_items (check quarantine if not found in main folder)
    order_items_base = os.path.join(gold_input, "silver_order_items", date)
    order_items_file = find_silver_file(order_items_base, "silver_order_items", date,
                                        quarantine_subfolder="quarantine_order_items")
    if order_items_file:
        data["order_items"] = spark.read.csv(order_items_file, header=True, inferSchema=True)
        logging.info(f"Loaded order_items from {order_items_file}")
    else:
        logging.warning(f"Missing Silver order_items file for date {date} in {order_items_base}")
        data["order_items"] = spark.createDataFrame([], schemas["order_items"])

    return data


def save_dataframe(df, prefix, date_str, timestamp, base_folder):
    """
    Save a DataFrame as a CSV file following the naming convention:
    {prefix}_{timestamp}.csv
    """
    temp_path = os.path.join(base_folder, f"{prefix}_{timestamp}_temp")
    final_path = os.path.join(base_folder, f"{prefix}_{timestamp}.csv")

    # Write to a temporary folder
    df.coalesce(1).write.csv(temp_path, header=True, mode="overwrite")

    # Move the CSV file from the temporary folder to the final destination
    for filename in os.listdir(temp_path):
        if filename.startswith("part-"):
            shutil.move(os.path.join(temp_path, filename), final_path)
            break

    # Remove the temporary folder
    shutil.rmtree(temp_path)


def gold_files_exist(date, timestamp, gold_output):
    """
    Check if Gold layer files already exist for the given date and timestamp.
    """
    fact_sales_folder = os.path.join(gold_output, "gold_fact_sales", date)
    fact_sales_simplified_folder = os.path.join(gold_output, "gold_fact_sales_simplified", date)

    fact_sales_file = os.path.join(fact_sales_folder, f"gold_fact_sales_{timestamp}.csv")
    fact_sales_simplified_file = os.path.join(fact_sales_simplified_folder, f"gold_fact_sales_simplified_{timestamp}.csv")

    return os.path.exists(fact_sales_file) or os.path.exists(fact_sales_simplified_file)


def process_gold_layer():
    """
    Process the Gold layer by joining data and applying business rules
    for each date found across the silver data entities.
    """
    # Get date folders from each silver entity and compute the union of dates
    dates_customers = set(get_date_folders_for_entity("silver_customers"))
    dates_orders = set(get_date_folders_for_entity("silver_orders"))
    dates_order_items = set(get_date_folders_for_entity("silver_order_items"))
    dates_products = set(get_date_folders_for_entity("silver_products"))
    all_dates = sorted(dates_customers.union(dates_orders).union(dates_order_items).union(dates_products))

    if not all_dates:
        logging.error("No Silver data found!")
        return

    for date in all_dates:
        logging.info(f"Processing data for date: {date}")

        # Load Silver data for the given date
        data = load_silver_data(date)
        silver_timestamp = data.get("silver_timestamp")
        if silver_timestamp is None:
            logging.error(
                f"Essential Silver timestamp missing for date: {date}. Skipping Gold processing for this date.")
            continue

        # Check if Gold files already exist for this date and timestamp
        if gold_files_exist(date, silver_timestamp, gold_output):
            logging.info(f"Gold files already exist for date {date} and timestamp {silver_timestamp}. Skipping processing.")
            continue

        logging.info(f"Orders loaded: {data['orders'].count()} rows")
        logging.info(f"Products loaded: {data['products'].count()} rows")
        logging.info(f"Order items loaded: {data['order_items'].count()} rows")
        logging.info(f"Customers loaded: {data['customers'].count()} rows")

        # Check for necessary tables before processing
        if (data["orders"] is None or data["orders"].rdd.isEmpty()) or \
                (data["products"] is None or data["products"].rdd.isEmpty()) or \
                (data["customers"] is None or data["customers"].rdd.isEmpty()):
            logging.error(f"Essential Silver tables missing or empty. Skipping Gold processing for date: {date}")
            continue

        # Business Rule 1: Exclude CANCELLED orders
        orders_filtered = data["orders"].filter(col("order_status") != "CANCELLED")
        logging.info(f"Orders after filtering CANCELLED status: {orders_filtered.count()} rows")

        # Business Rule 2: Date Validation
        orders_validated = orders_filtered.filter(col("order_date") >= "2020-01-01")
        logging.info(f"Orders after validating date: {orders_validated.count()} rows")

        if data["order_items"] is not None and not data["order_items"].rdd.isEmpty():
            # Drop duplicate columns from order_items and products so that we preserve orders' values.
            if "source_system" in data["order_items"].columns:
                data["order_items"] = data["order_items"].drop("source_system")
            if "ingest_timestamp" in data["order_items"].columns:
                data["order_items"] = data["order_items"].drop("ingest_timestamp")
            if "source_system" in data["products"].columns:
                data["products"] = data["products"].drop("source_system")
            if "ingest_timestamp" in data["products"].columns:
                data["products"] = data["products"].drop("ingest_timestamp")

            # Business Rule 3: Create fact_sales table with product details
            merged = orders_validated.join(data["order_items"], on="order_id", how="left_outer") \
                .join(data["products"], on="product_id", how="left_outer")
            logging.info(f"Merged data (with product details): {merged.count()} rows")

            # Add computed total
            merged = merged.withColumn("computed_total", col("quantity") * col("unit_price"))

            # If the merged data does not already have an ingest_timestamp column (from orders),
            # add one using the extracted silver_timestamp.
            if "ingest_timestamp" not in merged.columns:
                merged = merged.withColumn("ingest_timestamp", lit(silver_timestamp))
            # (Do not override source_system; the orders column remains as read from the silver file.)

            # Sum Check Validation
            mismatch = merged.filter(col("total_amount") != col("computed_total"))
            logging.info(f"Sum Check Mismatch: {mismatch.count()} rows")

            # Split into complete and simplified datasets
            fact_sales_data = merged.filter(
                col("order_id").isNotNull() &
                col("product_id").isNotNull() &
                col("quantity").isNotNull() &
                col("unit_price").isNotNull() &
                col("total_amount").isNotNull()
            )

            fact_sales_simplified_data = merged.filter(
                col("order_id").isNull() |
                col("product_id").isNull() |
                col("quantity").isNull() |
                col("unit_price").isNull() |
                col("total_amount").isNull()
            )

            # Save fact_sales data (complete records)
            if fact_sales_data.count() > 0:
                fact_sales_folder = os.path.join(gold_output, "gold_fact_sales", date)
                os.makedirs(fact_sales_folder, exist_ok=True)
                save_dataframe(fact_sales_data, "gold_fact_sales", date, silver_timestamp, fact_sales_folder)
                logging.info(f"Saved fact_sales table with product details to: {fact_sales_folder}")
                print(f"Generated gold_fact_sales_{silver_timestamp}.csv with {fact_sales_data.count()} rows")

            # Save fact_sales_simplified data (records with nulls)
            if fact_sales_simplified_data.count() > 0:
                fact_sales_simplified_folder = os.path.join(gold_output, "gold_fact_sales_simplified", date)
                os.makedirs(fact_sales_simplified_folder, exist_ok=True)
                save_dataframe(fact_sales_simplified_data, "gold_fact_sales_simplified", date, silver_timestamp,
                               fact_sales_simplified_folder)
                logging.info(f"Saved simplified fact_sales table to: {fact_sales_simplified_folder}")
                print(f"Generated gold_simplified_{silver_timestamp}.csv with {fact_sales_simplified_data.count()} rows")

            if mismatch.count() > 0:
                logging.warning(f"Sum Check Failed for {mismatch.count()} orders in date: {date}")

        else:
            # When order_items data is not available, process a simplified version.
            logging.info("Processing simplified version of fact_sales (without product details)...")
            # If orders_validated already has an ingest_timestamp from the silver file, keep it;
            # otherwise, add the extracted silver_timestamp.
            if "ingest_timestamp" not in orders_validated.columns:
                orders_validated = orders_validated.withColumn("ingest_timestamp", lit(silver_timestamp))

            fact_sales_simplified_data = orders_validated.filter(
                col("order_id").isNull() | col("total_amount").isNull()
            )
            fact_sales_valid_data = orders_validated.filter(
                col("order_id").isNotNull() & col("total_amount").isNotNull()
            )

            if fact_sales_valid_data.count() > 0:
                fact_sales_simplified_folder = os.path.join(gold_output, "gold_fact_sales_simplified", date)
                os.makedirs(fact_sales_simplified_folder, exist_ok=True)
                save_dataframe(fact_sales_valid_data, "gold_fact_sales_simplified", date, silver_timestamp,
                               fact_sales_simplified_folder)
                logging.info(f"Saved valid fact_sales simplified table to: {fact_sales_simplified_folder}")
                print(f"Generated gold_simplified_{silver_timestamp}.csv with no order_items data {fact_sales_valid_data.count()} rows")

            if fact_sales_simplified_data.count() > 0:
                logging.warning(f"Sum Check Failed for simplified fact_sales (due to null values) in date: {date}")

        logging.info(f"Completed processing for date: {date}")


if __name__ == "__main__":
    process_gold_layer()
