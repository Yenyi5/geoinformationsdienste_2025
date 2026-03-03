# 🗺️ STAC Geosearch Chat

STAC Geosearch Chat is an interactive application that allows users to query SpatioTemporal Asset Catalogs (STAC) for geospatial data using natural language. The app integrates a Large Language Model (LLM) to process user queries and provides corresponding results on a map. It also summarizes and evaluates the results in natural language and provides a list of similar collections.

[demo video]


## Prerequisites
1. **Python Environment**:
   - Python 3.8 or higher.
   - Install dependencies using `pip install -r requirements.txt`.
2. **Environment Variables**:
   - Create a `.env` file based on the `.env.template` provided.
   - Add your API keys and endpoints for the LLM and STAC catalogs.


## Starting the App
1. If you want, you can configure a `.env` file with the required API keys.
2. Run the app using Streamlit:
   ```bash
   streamlit run app.py
3. Open the app in your browser (default: http://localhost:8501).


## Using the App
1. Select a Catalog: Choose a STAC catalog to query. Currently, two catalogs are supported.
2. Enter LLM Settings: 
    - If not defined in your .env file, enter the API key, endpoint, and model name for your LLM provider.
3. Ask a Question: 
    - Enter a natural language query. Ensure your query includes:
        - A geographical place name.
        - A time range (if possible, some models can handle without).
    - Example: "Find Sentinel-2 data for Berlin in 2024."

## Adding New Catalogs
1. Run the collection_similarity.ipynb notebook to calculate similarity matrices for new catalogs.
2. Add the catalog URL to the app configuration.
⚠️ Currently, the notebook cannot handle empty collections.