import csv
import random
import datetime
import sys
import os
import json

DEFAULT_CONFIG_FILE = "config.json"

def load_config(config_file=DEFAULT_CONFIG_FILE):
    """
    Loads configuration from a JSON file. If it doesn't exist or is invalid,
    returns a default config dictionary.
    """
    default_config = {
        "paths": {
            "output_folder": ".",
            "tracker_file": "id_tracker.json"
        },
        "data_settings": {
            "num_rows_per_table": 500
        }
    }

    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                user_config = json.load(f)
            # Merge user_config into default_config (shallow merge)
            for key, value in user_config.items():
                if key in default_config and isinstance(value, dict):
                    # Merge sub-dict
                    default_config[key].update(value)
                else:
                    default_config[key] = value
        except (json.JSONDecodeError, IOError):
            print(f"Warning: Could not read/parse '{config_file}'. Using default config.")
    else:
        print(f"Warning: Config file '{config_file}' not found. Using default config.")

    return default_config

def load_id_tracker(tracker_path):
    """
    Load the last used IDs from a JSON file (tracker_path).
    If the file doesn't exist or is invalid, return default IDs.
    """
    if os.path.exists(tracker_path):
        try:
            with open(tracker_path, 'r') as f:
                data = json.load(f)
            return data
        except (json.JSONDecodeError, IOError):
            pass  # Fall back to default below

    return {
        "customers": 10000,
        "products": 10000,
        "orders": 10000,
        "order_items": 10000
    }

def save_id_tracker(tracker, tracker_path):
    """
    Save the updated IDs back to the given JSON file in readable format.
    """
    with open(tracker_path, 'w') as f:
        json.dump(tracker, f, indent=2)

def random_date(start_date, end_date):
    """Generate a random date between start_date and end_date."""
    start_ts = int(start_date.timestamp())
    end_ts = int(end_date.timestamp())
    rand_ts = random.randint(start_ts, end_ts)
    return datetime.date.fromtimestamp(rand_ts)

def generate_customers(num_rows, start_id):
    """Generate rows for 'customers' with strictly ascending IDs."""
    states = ["NY", "CA", "TX", "IL", "FL", "WA", "MA", "OH", "GA", "NC"]
    cities = [
        "New York", "Los Angeles", "Houston", "Chicago", "Miami",
        "Seattle", "Boston", "Cleveland", "Atlanta", "Charlotte"
    ]

    rows = []
    for i in range(start_id, start_id + num_rows):
        customer_id = i
        full_name = f"Customer {i}"
        email = f"customer{i}@example.com"
        address = f"{random.randint(1,9999)} {random.choice(['Main St','Oak Ave','Pine Rd','Maple Lane','1st Ave'])}"
        city = random.choice(cities)
        state = random.choice(states)
        rows.append([customer_id, full_name, email, address, city, state])
    return rows

def generate_products(num_rows, start_id):
    """Generate rows for 'products' with strictly ascending IDs."""
    categories = ["Electronics", "Wearables", "Home Decor", "Kitchen", "Sports"]

    rows = []
    for i in range(start_id, start_id + num_rows):
        product_id = i
        product_name = f"Product {i}"
        category = random.choice(categories)
        price = round(random.uniform(5.0, 1000.0), 2)
        rows.append([product_id, product_name, category, price])
    return rows

def generate_orders(num_rows, start_id, max_customer_id, order_date=None):
    """
    Generate rows for 'orders' with strictly ascending IDs.
    If 'order_date' is provided, all orders use that date; otherwise, pick random dates.
    """
    order_statuses = ["SHIPPED", "PENDING", "CANCELLED", "DELIVERED"]
    rows = []

    # If no date provided, pick random within this range
    start_range = datetime.date(2024, 1, 1)
    end_range = datetime.date(2025, 1, 1)

    for i in range(start_id, start_id + num_rows):
        order_id = i
        if order_date:
            od = order_date
        else:
            od = random_date(start_range, end_range)
        customer_id = random.randint(1, max_customer_id)
        total_amount = round(random.uniform(0.0, 5000.0), 2)
        status = random.choice(order_statuses)
        rows.append([order_id, od, customer_id, total_amount, status])
    return rows

def generate_order_items(num_rows, start_id, max_order_id, max_product_id):
    """Generate rows for 'order_items' with strictly ascending IDs."""
    rows = []
    for i in range(start_id, start_id + num_rows):
        order_item_id = i
        order_id = random.randint(1, max_order_id)
        product_id = random.randint(1, max_product_id)
        quantity = random.randint(1, 10)
        unit_price = round(random.uniform(5.0, 500.0), 2)
        rows.append([order_item_id, order_id, product_id, quantity, unit_price])
    return rows

def write_csv(file_name, header, rows):
    """Write the generated rows to a CSV file."""
    import csv
    with open(file_name, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

def main():
    """
    Steps:
      1) Load config (config.json) or use defaults.
      2) Load ID tracker (JSON).
      3) Determine 'order_date' from command line or default to today.
      4) Generate data with strictly ascending IDs.
      5) Save updated ID tracker.
      6) Write CSV files to the configured output folder with a second-level timestamp.
    """
    # -- 1. Load configuration
    config = load_config()
    output_folder = config["paths"]["output_folder"]
    tracker_file = config["paths"]["tracker_file"]
    num_rows = config["data_settings"]["num_rows_per_table"]

    # Make sure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # -- 2. Load or initialize ID tracker
    tracker = load_id_tracker(tracker_file)
    # tracker example structure:
    # {
    #   "customers": 10000,
    #   "products": 10000,
    #   "orders": 10000,
    #   "order_items": 10000
    # }

    # -- 3. Determine data date for orders
    if len(sys.argv) > 1:
        try:
            run_date = datetime.datetime.strptime(sys.argv[1], "%Y-%m-%d").date()
        except ValueError:
            print("Invalid date format. Use YYYY-MM-DD.")
            sys.exit(1)
    else:
        run_date = datetime.date.today()

    # Create a timestamp for the file names (to seconds level)
    file_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # -- 4. Generate data
    # Customers
    c_start_id = tracker["customers"] + 1
    customers_data = generate_customers(num_rows, c_start_id)
    tracker["customers"] = c_start_id + num_rows - 1

    # Products
    p_start_id = tracker["products"] + 1
    products_data = generate_products(num_rows, p_start_id)
    tracker["products"] = p_start_id + num_rows - 1

    # Orders
    o_start_id = tracker["orders"] + 1
    orders_data = generate_orders(
        num_rows=num_rows,
        start_id=o_start_id,
        max_customer_id=tracker["customers"],
        order_date=run_date
    )
    tracker["orders"] = o_start_id + num_rows - 1

    # Order Items
    oi_start_id = tracker["order_items"] + 1
    order_items_data = generate_order_items(
        num_rows=num_rows,
        start_id=oi_start_id,
        max_order_id=tracker["orders"],
        max_product_id=tracker["products"]
    )
    tracker["order_items"] = oi_start_id + num_rows - 1

    # -- 5. Save updated ID tracker
    save_id_tracker(tracker, tracker_file)

    # -- 6. Write CSV files to output_folder
    customers_file   = os.path.join(output_folder, f"raw_customers_{file_timestamp}.csv")
    products_file    = os.path.join(output_folder, f"raw_products_{file_timestamp}.csv")
    orders_file      = os.path.join(output_folder, f"raw_orders_{file_timestamp}.csv")
    order_items_file = os.path.join(output_folder, f"raw_order_items_{file_timestamp}.csv")

    write_csv(customers_file,
              ["customer_id", "full_name", "email", "address", "city", "state"],
              customers_data)

    write_csv(products_file,
              ["product_id", "product_name", "category", "price"],
              products_data)

    write_csv(orders_file,
              ["order_id", "order_date", "customer_id", "total_amount", "order_status"],
              orders_data)

    write_csv(order_items_file,
              ["order_item_id", "order_id", "product_id", "quantity", "unit_price"],
              order_items_data)

    print(f"\nFiles generated for data date: {run_date}")
    print(f"  {customers_file}")
    print(f"  {products_file}")
    print(f"  {orders_file}")
    print(f"  {order_items_file}")
    print("\nUpdated ID tracker (stored in):", tracker_file)
    print(json.dumps(tracker, indent=2))

if __name__ == "__main__":
    main()