import requests
import csv


def get_cellline_data(save_path):

    # URL to fetch data from
    url = "https://www.proteomicsdb.org/proteomicsdb/logic/api/tissuelist.xsodata/CA_AVAILABLEBIOLOGICALSOURCES_API?$select=TISSUE_ID,TISSUE_NAME,TISSUE_GROUP_NAME,TISSUE_CATEGORY,SCOPE_ID,SCOPE_NAME,QUANTIFICATION_METHOD_ID,QUANTIFICATION_METHOD_NAME,MS_LEVEL&$format=json"

    # Send a GET request to fetch the data
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        data = response.json()  # Parse JSON response
        results = data.get("d", {}).get("results", [])

        # Filter for "cell line" tissue category and extract unique tissue names, IDs, and category
        unique_tissues = {}
        for entry in results:
            if entry["TISSUE_CATEGORY"] == "cell line":
                tissue_id = entry["TISSUE_ID"]
                tissue_name = entry["TISSUE_NAME"]
                tissue_group = entry["TISSUE_GROUP_NAME"]
                unique_tissues[tissue_id] = (tissue_name, tissue_group, entry["TISSUE_CATEGORY"])

        # Write unique tissues to a CSV file
        with open(f"{save_path}proteomicsdb_cell_lines.csv", "w", newline="") as csvfile:
            csv_writer = csv.writer(csvfile)
            # Write header
            csv_writer.writerow(["TISSUE_ID", "TISSUE_NAME", "TISSUE_GROUP_NAME", "TISSUE_CATEGORY"])
            # Write each unique tissue entry
            for tissue_id, (tissue_name, tissue_group, tissue_category) in unique_tissues.items():
                csv_writer.writerow([tissue_id, tissue_name, tissue_group, tissue_category])

        print("Filtered data saved to 'filtered_tissues.csv'")
    else:
        print(f"Failed to fetch data. HTTP Status Code: {response.status_code}")
