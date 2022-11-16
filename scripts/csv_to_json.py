import argparse
import csv
import json
 
 
# Function to convert a CSV to JSON
# Takes the file paths as arguments
def make_json(csvFilePath, jsonFilePath, primaryKey):
     
    # create a dictionary
    data = {}
     
    # Open a csv reader called DictReader
    with open(csvFilePath, encoding='utf-8') as csvf:
        csvReader = csv.DictReader(csvf)
         
        # Convert each row into a dictionary
        # and add it to data
        for rows in csvReader:
            key = rows[primaryKey]
            data[key] = rows
 
    # Open a json writer, and use the json.dumps()
    # function to dump data
    with open(jsonFilePath, 'w', encoding='utf-8') as jsonf:
        jsonf.write(json.dumps(data, indent=4))
         
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c','--csv', type=str, required=True, default='import.csv')
    parser.add_argument('-j','--json', type=str, required=False)
    parser.add_argument('-k','--key', type=str, required=True, default='No')
    args = parser.parse_args()
    csv_path = args.csv
    if args.json is not None:
        json_path = args.json
    else:
        json_path = args.csv[:-3]+'json'
    primary_key = args.key
    # Call the make_json function
    make_json(csv_path, json_path, primary_key)