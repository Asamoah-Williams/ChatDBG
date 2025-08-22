import yaml
from pyprojroot import here
import os
from dotenv import load_dotenv

class LoadToolsConfig:
    def __init__(self):
        load_dotenv()
        with open(here("configs/tools_config.yml")) as cfg:
            app_config = yaml.load(cfg, Loader=yaml.FullLoader)

        # set environment variables 
        os.environ['OPENAI_API_KEY'] = os.getenv("OPEN_AI_API_KEY")
        os.environ["CHROMA_API_KEY"] = os.getenv("CHROMA_API_KEY")
        os.environ["CHROMA_TENANT"] = os.getenv("CHROMA_TENANT")
        os.environ["CHROMA_DATABASE"] = os.getenv("CHROMA_DATABASE")
        os.environ["SQLDB_UID"] = os.getenv("SQLDB_UID")
        os.environ["SQLDB_PASSWORD"] = os.getenv("SQLDB_PASSWORD")

        # Primary agent
        self.primary_agent_llm = app_config["primary_agent"]["llm"]
        self.primary_agent_llm_temperature = app_config["primary_agent"]["llm_temperature"]
        
        # SQL Agent configs 
        self.sqlagent_llm = app_config["sqlagent_configs"]["llm"]
        self.sqlagent_llm_temperature = float(
            app_config["sqlagent_configs"]["llm_temperature"])
        
        # GuideRAGTool configs 
        self.guiderag_collection_name = app_config["guiderag_configs"]["collection_name"]
        self.guiderag_llm = app_config["guiderag_configs"]["llm"]
        self.guiderag_llm_temperature = app_config["guiderag_configs"]["llm_temperature"]
        self.guiderag_embedding_model = app_config["guiderag_configs"]["embedding_model"]
        self.guiderag_chunk_size = app_config["guiderag_configs"]["chunk_size"]
        self.guiderag_chunk_overlap = app_config["guiderag_configs"]["chunk_overlap"]
        self.guiderag_k = app_config["guiderag_configs"]["k"]

        # TavilySearchTool configs
        self.tavily_search_max_results = app_config["tavily_configs"]["max_results"]
