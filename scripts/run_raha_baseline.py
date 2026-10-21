"""
Baseline Execution Script: Raha on Hospital Dataset (B3 Error Detection Baseline)
Evaluates Raha semi-supervised detection with a 20-labeled-tuple budget.
Includes Windows multiprocessing file-lock patch for dBoost temp files.
"""
import os
import sys
import random
import time
import json
import hashlib
import tempfile
import pickle
import re
import numpy as np
import pandas
import raha
import raha.tools.dBoost.dboost.imported_dboost
import raha.tools.KATARA.katara

def patched_strategy_runner_process(self, args):
    """
    Robust strategy runner with Windows safe-file-deletion handling.
    """
    d, algorithm, configuration = args
    start_time = time.time()
    strategy_name = json.dumps([algorithm, configuration])
    strategy_name_hash = str(int(hashlib.sha1(strategy_name.encode("utf-8")).hexdigest(), 16))
    outputted_cells = {}
    
    if algorithm == "OD":
        dataset_path = os.path.join(tempfile.gettempdir(), d.name + "-" + strategy_name_hash + ".csv")
        d.write_csv_dataset(dataset_path, d.dataframe)
        params = ["-F", ",", "--statistical", "0.5"] + ["--" + configuration[0]] + configuration[1:] + [dataset_path]
        try:
            raha.tools.dBoost.dboost.imported_dboost.run(params)
        except Exception as e:
            pass
        algorithm_results_path = dataset_path + "-dboost_output.csv"
        if os.path.exists(algorithm_results_path):
            try:
                ocdf = pandas.read_csv(algorithm_results_path, sep=",", header=None, encoding="utf-8", dtype=str,
                                       keep_default_na=False, low_memory=False).apply(lambda x: x.str.strip())
                for i, j in ocdf.values.tolist():
                    if int(i) > 0:
                        outputted_cells[(int(i) - 1, int(j))] = ""
            except Exception:
                pass
            try:
                os.remove(algorithm_results_path)
            except Exception:
                pass
        try:
            os.remove(dataset_path)
        except Exception:
            pass
            
    elif algorithm == "PVD":
        attribute, ch = configuration
        j = d.dataframe.columns.get_loc(attribute)
        for i, value in d.dataframe[attribute].items():
            try:
                if len(re.findall("[" + ch + "]", value, re.UNICODE)) > 0:
                    outputted_cells[(i, j)] = ""
            except Exception:
                continue
                
    elif algorithm == "RVD":
        l_attribute, r_attribute = configuration
        l_j = d.dataframe.columns.get_loc(l_attribute)
        r_j = d.dataframe.columns.get_loc(r_attribute)
        value_dictionary = {}
        for i, row in d.dataframe.iterrows():
            if row[l_attribute]:
                if row[l_attribute] not in value_dictionary:
                    value_dictionary[row[l_attribute]] = {}
                if row[r_attribute]:
                    value_dictionary[row[l_attribute]][row[r_attribute]] = 1
        for i, row in d.dataframe.iterrows():
            if row[l_attribute] in value_dictionary and len(value_dictionary[row[l_attribute]]) > 1:
                outputted_cells[(i, l_j)] = ""
                outputted_cells[(i, r_j)] = ""
                
    elif algorithm == "KBVD":
        try:
            outputted_cells = raha.tools.KATARA.katara.run(d, configuration)
        except Exception:
            outputted_cells = {}
            
    detected_cells_list = list(outputted_cells.keys())
    strategy_profile = {
        "name": strategy_name,
        "output": detected_cells_list,
        "runtime": time.time() - start_time
    }
    if self.SAVE_RESULTS:
        os.makedirs(os.path.join(d.results_folder, "strategy-profiling"), exist_ok=True)
        pickle.dump(strategy_profile, open(os.path.join(d.results_folder, "strategy-profiling",
                                                        strategy_name_hash + ".dictionary"), "wb"))
    if self.VERBOSE:
        print("{} cells are detected by {}.".format(len(detected_cells_list), strategy_name))
    return strategy_profile

# Monkeypatch method
raha.detection.Detection._strategy_runner_process = patched_strategy_runner_process

def run_raha_hospital(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raha_root = os.path.join(base_dir, "data", "raw", "raha")
    folder = os.path.join(raha_root, "datasets", "hospital")  # layout of a Raha clone
    if not os.path.isdir(folder):
        folder = os.path.join(raha_root, "hospital")
    hospital_dirty = os.path.join(folder, "dirty.csv")
    hospital_clean = os.path.join(folder, "clean.csv")
    
    dataset_dict = {
        "name": "hospital",
        "path": hospital_dirty,
        "clean_path": hospital_clean
    }
    
    print(f"Initializing Raha Detection for {dataset_dict['name']} (Seed: {seed})...")
    app = raha.Detection()
    app.LABELING_BUDGET = 20
    app.VERBOSE = False  # Keep terminal output clean
    
    detection_dictionary = app.run(dataset_dict)
    
    d = raha.dataset.Dataset(dataset_dict)
    p, r, f = d.get_data_cleaning_evaluation(detection_dictionary)[:3]
    
    print("\n" + "="*50)
    print(f"Raha Benchmark Results on Hospital (20 Labels, Seed {seed}):")
    print(f"Precision: {p:.4f}")
    print(f"Recall:    {r:.4f}")
    print(f"F1-Score:  {f:.4f}")
    print("="*50)
    return p, r, f

if __name__ == "__main__":
    run_raha_hospital(seed=42)
