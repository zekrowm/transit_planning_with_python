# Transit Planning With Python

Welcome to **Transit Planning With Python**, a toolkit of off‑the‑shelf Python scripts designed for data‑driven transit planning at medium‑sized agencies. Each tool focuses on accomplishing a common task or solving a common challenge in transit analysis and planning. The most common input is static ***General Transit Feed Specification (GTFS)*** data, but ridership data, operational data, Census data, and road network data are also commonly used.

## 🚀 Features

Our scripts automate complex transit analysis. The current modules and data types supported include:

Data Preparation and Visualization

- ***Ridership Maps:*** Automate the processing of ridership, stop, and area data for use in heatmaps.
- ***Choropleth Data:*** Generate ready-to-map data for ridership trends and demographic overlays.

Operational Analysis

- ***Bus Bay Conflict Optimization:*** Identify scheduling conflicts and suggest improved assignments to reduce them.
- ***Headway and Schedule Analysis:*** Automatically calculate headways, span, and trip frequencies for any schedule.

Fieldwork Tools

- ***On-Site Checklists:*** Create field checklists for monitoring on-time performance and ridership at stops or while riding routes.

Network Analysis

- ***Route Identification:*** Quickly find transit routes operating near a defined point.
- ***Isochrone and Accessibility*** Mapping: Build transit and roadway networks to analyze travel times and service coverage.

GTFS Validation and Export

- ***GTFS Location and Name Validation:*** Cross-check GTFS stop and route data against standard shapefiles like roadway_centerlines.shp and bus_routes.shp.
- ***Schedule Export:*** Export transit schedules directly to Excel (.xlsx) format for sharing and reporting.

Demographic Analysis

- ***Census Integration:*** Process and combine Census tables with shapefiles for demographic insights.
- ***Service Population Analysis:*** Calculate residential and employee populations served by individual transit routes and systems.

## 📂 Repository Structure

The repository is organized for ease of use, with:

- ***Standalone scripts:*** Each tool is fully documented with comments explaining field name assumptions, file formats, and usage instructions.
- ***Standard data formats:*** Most scripts are designed to work with commonly used data types like GTFS files and shapefiles. Specific requirements are outlined in the script comments.
- ***Sample Data:*** A synthetic GTFS dataset is provided in `sample_data/gtfs` for testing and experimentation.

## 🛠️ Requirements

- Python 3.9+
- Common libraries like pandas, geopandas, rapidfuzz, networkx, and others listed in requirements.txt.

## 🧑‍💻 How to Use

The **transit_planning_with_python** tools are designed to work on most systems with Python installed. Here are some key considerations based on your setup:
1. **Work PC with ArcPro Installed:**
   - If ArcPro is installed, libraries such as `arcpy` and other useful dependencies are already included.
   - However, your organization may restrict the installation of additional libraries like `geopandas` or `rapidfuzz`.
   - If unrestricted, note that `geopandas` conflicts with `arcpy`, so you will need to create a separate Python environment to use it.
2. **Home Computer with Python Installed:**
   - On a personal system, you can install Python and any libraries using `pip` without organization restrictions.
   - Keep in mind that `arcpy` is unavailable outside of ArcPro/ArcMap environments, so certain features relying on `arcpy` won't work.

Where possible, we will provide both `arcpy` and `geopandas` versions of geospatial scripts to accomodate these different setups.

---

### 🖥️ Option A: Using Python on a Work Computer with ArcPro

1. **Open a Notebook**
   - If ArcPro is installed on your computer, then Python is as well.
   - You can launch ArcPro and then open, create, or save a Notebook file (.ipynb) within that program.
   - You can also find "Jupyter Notebook" like any other program on your computer. Clicking it will open your default browser with the notebook interface.
   - Alternately, you can run this command in the Command Prompt to launch Jupyter Notebook:
     ```bash
     jupyter notebook
     ```

2. **Get the Script(s) You Need**
   - Navigate to scripts that are useful to you and your agency. Then copy and paste their contents into an active notebook file or download them        as .py files. You do not need a GitHub account to do this.
   - Alternately, you can clone or download the whole repository from GitHub.
 
3. **Run the Script**
   - Confirm that desired code is present in notebook file and locate "CONFIGURATION" section.
   - "Update file paths:" At a minimum, you will need to update the folder and file paths to point to your data and specify where to save any
     output. There may also be additional configuration choices (e.g. choice of CRS, list of routes or stops to analyze)
   - You do not need to modify any code outside of the configuration section, which contains the core script logic.
   - Run the script, follow any printed instructions or error messages, and check the output for reasonableness and/or accuracy.

---

### ❓ What is Jupyter Notebook?

Jupyter Notebook is a powerful tool for running Python scripts in an interactive environment that opens in your web browser (or within ArcPro). It allows you to write, test, and visualize Python code in a user-friendly way.

---

### 🏡 Option B: Setting Up Python on a Home Computer

1. **Download and Install Python**
   - Visit the [official Python website](https://www.python.org/downloads/) and download Python 3.9 or later.
   - During installation, ensure you check the option to **"Add Python to PATH"**. This step is crucial for the command-line tools to work correctly.

2. **Install JupyterLab and Required Libraries**
   - Open the Command Prompt (search for "cmd" in your Start menu) and run the following command:
     ```bash
     pip install jupyterlab pandas geopandas shapely matplotlib networkx openpyxl rapidfuzz pulp
     ```
   - Wait for the installation to complete. If you see warnings about scripts not being on the PATH, don't worry - you can still use these tools.

3. **Launch JupyterLab**
   - After installation, search for "Command Prompt" and open it like any other program on your computer.
   - From the Command Prompt, type the following into the Command command and press Enter:
     ```bash
     jupyter lab
     ```
   - Once you have JupyterLab open in your web browser, you can browse your local files and create new notebook files.

4. **Get the Script(s) You Need**
   - Navigate to scripts that are useful to you and your agency. Then copy and paste their contents into an active notebook file or download them as .py files. You do not need a GitHub account to do this.
   - Alternately, you can clone or download the whole repository from GitHub.
 
5. **Run the Script**
   - Confirm that desired code is present in notebook file and locate "CONFIGURATION" section.
   - "Update file paths:" At a minimum, you will need to update the folder and file paths to point to your data and specify where to save any
     output. There may also be additional configuration choices (e.g. choice of CRS, list of routes or stops to analyze)
   - You do not need to modify any code outside of the configuration section, which contains the core script logic.
   - Run the script, follow any printed instructions or error messages, and check the output for reasonableness and/or accuracy.

---

## 💡 Notes for Beginners

If you encounter any issues while installing Python or libraries, or if JupyterLab doesn't open as expected, here are some common troubleshooting steps:

1. **Verify Python Installation**

   Open the Command Prompt and type:
     ```bash
     python --version
     ```

   If this command doesn't show a Python version (e.g., `Python 3.11.5`), Python may not be installed correctly, or it isn't added to your PATH. In that case:
   - Reinstall Python from the [official website](https://www.python.org/downloads/).
   - During installation, ensure the option **"Add Python to PATH"** is checked.
   - Or proceed to **Step 3: Fixing PATH Issues**

2. **Check JupyterLab Installation**

   To confirm JupyterLab is installed, type:
     ```bash
     jupyter lab --version
     ```

   If this fails with an error like `'jupyter' is not recognized as an internal or external command`, JupyterLab is either not installed or the PATH variable isn't set correctly.  
   Try reinstalling JupyterLab by running:
     ```bash
     pip install jupyterlab
     ```

   If the issue persists, follow the steps in **Fixing PATH Issues** below.

3. **Fixing PATH Issues**

   If Python or Jupyter commands aren't recognized, you may need to manually add the Python Scripts directory to your PATH:

   - Locate the directory where Python installed its scripts. For most users, this will be:
        ```makefile
     C:\Users\<YourUsername>\AppData\Roaming\Python\Python311\Scripts
     ```

   - Add this directory to your PATH:
      - Press <kbd>Win + S</kbd>, search for **Environment Variables**, and open **Edit the system environment variables**.
      - In the dialog, click **Environment Variables**.
      - Under **User variables**, find and edit the **Path** variable.
      - Add the Scripts directory to the list by clicking **New** and pasting the path above.

   - Restart your Command Prompt and verify by typing:
     ```bash
     jupyter lab --version
     ```   

4. **Reinstalling or Updating Pip**

   If the installation commands fail with errors like `pip is not recognized` or `ModuleNotFoundError`, try reinstalling or upgrading pip:
     ```bash
     python -m ensurepip --upgrade
     python -m pip install --upgrade pip
     ```   

5. **Alternative Way to Open JupyterLab**

   Instead of relying on PATH, you can directly use Python to run JupyterLab:
     ```bash
     python -m jupyter lab
     ```
     
   This bypasses PATH issues entirely.

By following these steps, you should have Python and the necessary tools ready to run your scripts efficiently. If you continue to face issues, feel free to consult the official documentation for [Python](https://docs.python.org/3/) or [JupyterLab](https://jupyterlab.readthedocs.io/en/latest/).

---

### ❓ What is JupyterLab?

JupyterLab is an advanced interactive development environment for data science and computational research that runs in your web browser. It builds on the functionality of Jupyter Notebook, providing a flexible interface where you can write, test, and visualize Python code alongside other tasks such as file browsing, markdown editing, and interactive data exploration.

---

### 🌟 Tips for Success

- **File Paths:** Ensure all file paths are updated in the scripts to match your local system.
- **Permissions:** If IT restrictions block library installations on your work computer, setting up Python on a home computer is recommended.
- **Sample Data:** If you want to try the scripts but don't have your own data yet, you can use the sample data provided in `sample_data/gtfs`. This folder contains a synthetic "Toy Transit" agency feed.

  Example of loading the sample data using `helpers.gtfs_helpers`:

  ```python
  from helpers.gtfs_helpers import load_gtfs_data

  # Point to the sample data folder
  gtfs_path = "sample_data/gtfs"

  # Load the specific files present in the sample feed
  data = load_gtfs_data(
      gtfs_path,
      files=("agency.txt", "stops.txt", "routes.txt", "trips.txt", "stop_times.txt", "calendar.txt", "shapes.txt")
  )

  print(data["routes"].head())
  ```

## 🤝 Contributing

We welcome your contributions! Feel free to open an issue or submit a pull request.

## 📄 License

This repository is licensed under the [GNU General Public License v3.0](https://choosealicense.com/licenses/gpl-3.0/).
You are free to use, modify, and distribute the tools within this repository, as long as you adhere to the license terms.

We hope these tools speed up and simplify your transit planning workflows! If you encounter issues or have questions, feel free to open an issue or contact us. 🚍
