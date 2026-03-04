# STAC Geosearch Chat

STAC Geosearch Chat is an interactive application that allows users to query SpatioTemporal Asset Catalogs (STAC) for geospatial data using natural language. The app integrates a Large Language Model (LLM) to process user queries and provides corresponding results on a map. It also summarizes and evaluates the results in natural language and provides a list of similar collections. New catalogs to query can be integrated, but a similarity matrix must be calculated to get similar collections recommended. 

[![Watch the demo](assets/demo_preview.png)] (https://datashare.tu-dresden.de/s/J6xcmMD9PxwwGSb)


## Prerequisites
1. **Python Environment**:
   - Python 3.8 or higher.
   - Install dependencies using `pip install -r requirements.txt`.
2. **API keys**:
   - LLM API key (OpenAI compatible provider): [OpenAI](https://openai.com/de-DE/api/) and [AcademicCloud](https://docs.hpc.gwdg.de/services/saia/index.html#api-request) implemented
   - [LangSmith](https://www.langchain.com/) API key for tracing & debugging LangChain workflows
   - Optionally: Add your API keys and endpoints for the LLM and STAC catalogs to a .env file.


## Starting the App
1. If you want, you can configure a `.env` file with the required API keys.
   ```bash
   cp .env.template .env
3. Run the app using Streamlit:
   ```bash
   streamlit run app.py
4. Open the app in your browser (default: http://localhost:8501).


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
1. Run the collection_similarity.ipynb notebook to calculate similarity matrices for new catalog.
2. Add the catalog URL to the app configuration.

⚠️ Currently, the notebook cannot handle empty collections.
