import os
import yaml
import logging
import zipfile
import numpy as np
import pandas as pd
import traceback
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime
import argparse

# Model imports
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from hdbscan import HDBSCAN
import gensim
import gensim.corpora as corpora
from gensim.models.coherencemodel import CoherenceModel

# Dash imports
import dash
from dash import dcc, html, dash_table
import plotly.express as px
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("topic_modeling.log")
    ]
)
logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self, config: Dict):
        self.input_filepath = config["input_filepath"]
        self.review_column = config.get("review_column", "review")
        self.timestamp_column = config.get("timestamp_column", "yearMonth")
        self.sample_size = config.get("sample_size", None)
    
    def load_reviews(self) -> pd.DataFrame:
        logger.info(f"Loading data from {self.input_filepath}")
        try:
            df = pd.read_csv(self.input_filepath)
            logger.info(f"Total documents loaded: {len(df)}")
            
            if self.review_column not in df.columns:
                available_cols = ", ".join(df.columns)
                logger.error(f"Review column '{self.review_column}' not found. Available columns: {available_cols}")
                raise ValueError(f"Review column '{self.review_column}' not found in dataset")
            
            if self.timestamp_column not in df.columns:
                available_cols = ", ".join(df.columns)
                logger.error(f"Timestamp column '{self.timestamp_column}' not found. Available columns: {available_cols}")
                raise ValueError(f"Timestamp column '{self.timestamp_column}' not found in dataset")
            
            missing_reviews = df[self.review_column].isna().sum()
            logger.info(f"Missing values in review column: {missing_reviews} ({missing_reviews/len(df):.2%})")
            
            df_cleaned = df.dropna(subset=[self.review_column, self.timestamp_column])
            logger.info(f"Documents after removing missing values: {len(df_cleaned)}")
            
            df_cleaned[self.review_column] = df_cleaned[self.review_column].apply(lambda x: str(x).strip())
            df_cleaned = df_cleaned[df_cleaned[self.review_column].str.len() > 0]
            logger.info(f"Final document count after cleaning: {len(df_cleaned)}")
            
            if self.sample_size is not None and self.sample_size < len(df_cleaned):
                logger.info(f"Sampling {self.sample_size} documents")
                df_cleaned = df_cleaned.sample(self.sample_size, random_state=42)
            
            result_df = df_cleaned[[self.review_column, self.timestamp_column]]
            result_df.columns = ["review", "timestamp"] # +++
            logger.info(f"Returning DataFrame with {len(result_df)} rows and columns: {list(result_df.columns)}")
            
            return result_df
        
        except FileNotFoundError:
            logger.error(f"File not found: {self.input_filepath}")
            raise
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

class BERTopicConfigurator:
    def __init__(self, config: Dict):
        logger.info("Configuring BERTopic model")
        self.config = config
        
        try:
            self.embedding_model = SentenceTransformer(self.config["transformer_name"])
            logger.info(f"Initialized embedding model: {self.config['transformer_name']}")
        except Exception as e:
            logger.error(f"Error initializing embedding model: {str(e)}")
            raise
        
        self.umap_model = UMAP(**self.config["umap"])
        logger.info(f"Initialized UMAP with parameters: {self.config['umap']}")
        
        umap_2d_params = self.config["umap"].copy()
        umap_2d_params["n_components"] = 2
        self.umap_model_2d = UMAP(**umap_2d_params)
        logger.info(f"Initialized UMAP with parameters: {umap_2d_params}")
        
        self.hdbscan_model = HDBSCAN(**self.config["hdbscan"])
        logger.info(f"Initialized HDBSCAN with parameters: {self.config['hdbscan']}")
        
        self.vectorizer_model = CountVectorizer(
            stop_words=None,
            ngram_range=tuple(self.config["vectorizer"]["ngram_range"])
        )
        logger.info(f"Initialized CountVectorizer with parameters: {self.config['vectorizer']}")
        
        self.topic_model = BERTopic(
            min_topic_size=self.config["min_topic_size"],
            embedding_model=self.embedding_model,
            umap_model=self.umap_model,
            hdbscan_model=self.hdbscan_model,
            vectorizer_model=self.vectorizer_model,
            language=self.config["language"],
            nr_topics=self.config["nr_topics"],
            verbose=True
        )
        logger.info("BERTopic model configuration complete")

class TopicModeler:
    def __init__(self, topic_model):
        self.topic_model = topic_model
        self.topics = None
        self.probs = None
        self.doc_info = None
        self.embeddings = None
        self.reduced_embeddings = None
    
    def fit(self, documents: List[str]) -> Tuple[List[int], np.ndarray, np.ndarray, np.ndarray]:
        logger.info("Training BERTopic model")
        if not documents or not isinstance(documents, list):
            raise ValueError("Documents must be a non-empty list of strings.")
        
        try:
            self.embeddings = self.topic_model.embedding_model.encode(documents)
            self.topics, self.probs = self.topic_model.fit_transform(
                documents=documents, 
                embeddings=self.embeddings
            )
            
            info = self.topic_model.get_topic_info()
            logger.info(f"Number of topics identified: {len(info)}")
            logger.info(f"Distribution: {info['Count'].value_counts().to_dict()}")
            logger.info(f"Topic assignments: {self.topics[:3]}")
            logger.info(f"Topic probabilities: {self.probs[:3]}")
            
            self.doc_info = pd.DataFrame({
                'Document': documents,
                'Topic': self.topics,
                'Probability': self.probs
            })
            
            if hasattr(self.topic_model, "umap_model"):
                self.reduced_embeddings = self.topic_model.umap_model.transform(self.embeddings)
            else:
                raise AttributeError("The BERTopic model does not have a UMAP model for dimensionality reduction.")
                
            return self.topics, self.probs, self.embeddings, self.reduced_embeddings
            
        except Exception as e:
            logger.error(f"Error during model training: {str(e)}\n{traceback.format_exc()}")
            raise
    
    def save(self, output_dir: str) -> str:
        if not os.path.exists(output_dir):
            raise FileNotFoundError(f"The output directory '{output_dir}' does not exist.")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(output_dir, f"bertopic_model_{timestamp}")
        os.makedirs(save_path, exist_ok=True)
        logger.info(f"Saving model to {save_path}")
        
        try:
            embedding_model = self.topic_model.embedding_model
            self.topic_model.save(
                os.path.join(save_path, "model"),
                serialization="safetensors",
                save_ctfidf=True,
                save_embedding_model=embedding_model
            )
            
            if self.doc_info is not None:
                self.doc_info.to_csv(os.path.join(save_path, "document_topics.csv"), index=False)
            
            topic_info = self.topic_model.get_topic_info()
            topic_info.to_csv(os.path.join(save_path, "topic_info.csv"), index=False)
            
            topic_terms = {}
            for topic in topic_info['Topic'].unique():
                if topic != -1:
                    terms = self.topic_model.get_topic(topic)
                    topic_terms[topic] = [term[0] for term in terms]
            pd.DataFrame(topic_terms).to_csv(os.path.join(save_path, "topic_terms.csv"))
            
            if self.reduced_embeddings is not None:
                np.save(os.path.join(save_path, "reduced_embeddings.npy"), self.reduced_embeddings)
            
            logger.info(f"Model and artifacts saved successfully to {save_path}")
            return save_path
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}\n{traceback.format_exc()}")
            raise

class Evaluator:
    def __init__(self, topic_model: BERTopic, config: Dict):
        self.topic_model = topic_model
        self.config = config
        self.coherence_metrics = config.get("coherence_metrics", ["c_v"])
    
    def calculate_coherence(self, documents: List[str]) -> Dict[str, float]:
        logger.info("Calculating coherence scores using Gensim")
        scores = {}
        
        try:
            logger.info("Tokenizing documents for coherence calculation")
            tokenized_texts = [doc.split() for doc in documents if doc]
            
            logger.info("Extracting topic words from model")
            topics_dict = self.topic_model.get_topics()
            if -1 in topics_dict:
                del topics_dict[-1]
                
            gensim_topics = []
            for topic_id in sorted(topics_dict.keys()):
                topic_words = [word for word, _ in topics_dict[topic_id]]
                gensim_topics.append(topic_words)
            
            logger.info("Creating Gensim dictionary")
            dictionary = corpora.Dictionary(tokenized_texts)
            
            for metric in self.coherence_metrics:
                try:
                    logger.info(f"Calculating {metric} coherence")
                    cm = CoherenceModel(
                        topics=gensim_topics,
                        texts=tokenized_texts,
                        dictionary=dictionary,
                        coherence=metric
                    )
                    score = cm.get_coherence()
                    scores[metric] = score
                    logger.info(f"{metric} coherence: {score:.4f}")
                except Exception as e:
                    logger.error(f"Error calculating {metric} coherence: {str(e)}")
                    scores[metric] = None
            
            logger.info(f"Coherence scores: {scores}")
            return scores
            
        except Exception as e:
            logger.error(f"Error in coherence calculation pipeline: {str(e)}")
            return scores
    
    def calculate_topic_diversity(self) -> float:
        try:
            topics = self.topic_model.get_topics()
            if -1 in topics:
                del topics[-1]
                
            if not topics:
                logger.warning("No topics found for diversity calculation")
                return 0.0
                
            unique_words = set()
            for topic in topics.values():
                unique_words.update([word for word, _ in topic])
                
            total_words = sum(len(topic) for topic in topics.values())
            diversity = len(unique_words) / total_words if total_words > 0 else 0
            
            logger.info(f"Topic diversity: {diversity:.4f}")
            return diversity
            
        except Exception as e:
            logger.error(f"Error calculating topic diversity: {str(e)}")
            return 0.0
    
    def calculate_metrics(self, documents: List[str]) -> Dict[str, float]:
        coherence_scores = self.calculate_coherence(documents)
        topic_diversity = self.calculate_topic_diversity()
        
        return {
            "coherence_scores": coherence_scores,
            "topic_diversity": topic_diversity
        }

# Visualization utilities
def visualize_topics_2d(topics, reduced_embeddings, docs, width=1200, height=900):
    """Create a 2D visualization of topic distribution"""
    if reduced_embeddings.shape[1] < 2:
        logger.error("Reduced embeddings must have at least two dimensions")
        raise ValueError("Reduced embeddings must have at least two dimensions")
    
    df = pd.DataFrame({
        'x': reduced_embeddings[:, 0],
        'y': reduced_embeddings[:, 1],
        'Topic': [f"Topic {t}" if t != -1 else "Outliers (-1)" for t in topics],
        'Review': [doc[:100] + "..." if len(doc) > 100 else doc for doc in docs]
    })
    
    fig = px.scatter(
        df,
        x='x',
        y='y',
        color='Topic',
        hover_data={
            'Review': True, 
            'Topic': True, 
            'x': False, 
            'y': False
        },
        title="Topic Visualization in 2D Space",
        labels={'x': 'Component 1', 'y': 'Component 2'},
        width=width,
        height=height
    )
    
    fig.update_traces(marker=dict(size=6, opacity=0.7))
    fig.update_layout(
        legend_title_text='Topics',
        showlegend=True,
        title_x=0.5,
        hovermode='closest',
        legend=dict(
            yanchor="middle",
            y=0.50,
            xanchor="right",
            x=1.15
        )
    )
    
    return fig

# Dashboard components
class DashboardComponents:
    @staticmethod
    def create_model_performance_section(coherence_scores, topic_diversity, topic_info):
        """Create the model performance section with coherence scores and topic diversity"""
        return html.Div([
            html.H2("Model Performance", className="text-xl font-bold mb-3"),
            html.Div([
                # Coherence Scores Table
                html.Div([
                    html.H3("Coherence Scores", className="text-lg font-semibold mb-2"),
                    dash_table.DataTable(
                        id="coherence-table",
                        columns=[
                            {"name": "Metric", "id": "Metric"},
                            {"name": "Score", "id": "Score"}
                        ],
                        data=[
                            {"Metric": metric, "Score": f"{score:.4f}" if score is not None else "N/A"} 
                            for metric, score in coherence_scores.items()
                        ],
                        style_table={"overflowX": "auto"},
                        style_cell={"textAlign": "left", "padding": "10px"},
                        style_header={"backgroundColor": "rgb(230, 230, 230)", "fontWeight": "bold"}
                    )
                ], className="p-4 border rounded bg-white mb-4"),
                
                # Topic Diversity Card
                html.Div([
                    html.H3("Topic Diversity", className="text-lg font-semibold mb-2"),
                    html.Div([
                        html.Span(f"{topic_diversity:.4f}", className="text-2xl font-bold"),
                        html.Span(" / 1.0", className="text-gray-500")
                    ], className="flex items-center"),
                    html.P("Higher scores indicate more diverse topics", 
                           className="text-sm text-gray-600 mt-1")
                ], className="p-4 border rounded bg-white mb-4"),
                
                # Topic Summary Table
                html.Div([
                    html.H3("Topic Summary", className="text-lg font-semibold mb-2"),
                    dash_table.DataTable(
                        id="topic-summary",
                        columns=[
                            {"name": "Topic", "id": "Topic"},
                            {"name": "Count", "id": "Count"},
                            {"name": "Name", "id": "Name"},
                        ],
                        data=topic_info[["Topic", "Count", "Name"]].to_dict("records"),
                        style_table={"overflowX": "auto"},
                        style_cell={"textAlign": "left", "padding": "10px"},
                        style_header={"backgroundColor": "rgb(230, 230, 230)", "fontWeight": "bold"},
                        page_size=10
                    )
                ], className="p-4 border rounded bg-white"),
            ])
        ], className="w-full lg:w-1/3 p-4")
    
    @staticmethod
    def create_topic_visualizations(topic_model, topics, reduced_embeddings, docs, topics_over_time):
        """Create the topic visualizations section with various visualization options"""
        return html.Div([
            html.H2("Topic Visualizations", className="text-xl font-bold mb-3"),
            dcc.Tabs([
                # Topic Overview Tab
                dcc.Tab(
                    label="Topic Overview", 
                    children=[
                        dcc.Graph(
                            id="topics-overview",
                            figure=topic_model.visualize_topics(),
                            style={"height": "600px"}
                        )
                    ], 
                    className="p-4"
                ),
                
                # Topic Terms Tab
                dcc.Tab(
                    label="Topic Terms", 
                    children=[
                        dcc.Graph(
                            id="barchart-graph",
                            figure=topic_model.visualize_barchart(top_n_topics=10),
                            style={"height": "600px"}
                        )
                    ], 
                    className="p-4"
                ),
                
                # Similarity Matrix Tab
                dcc.Tab(
                    label="Topic Similarity", 
                    children=[
                        dcc.Graph(
                            id="similarity-matrix",
                            figure=topic_model.visualize_heatmap(),
                            style={"height": "600px"}
                        )
                    ], 
                    className="p-4"
                ),
                
                # Document Map Tab
                dcc.Tab(
                    label="Document Map", 
                    children=[
                        dcc.Graph(
                            id="documents-map",
                            figure=visualize_topics_2d(
                                topics=topics,
                                reduced_embeddings=reduced_embeddings,
                                docs=docs
                            ),
                            style={"height": "600px"}
                        )
                    ], 
                    className="p-4"
                ),
                
                # Topics Over Time Tab
                dcc.Tab(
                    label="Topic Over Time", 
                    children=[
                        dcc.Graph(
                            id="topic-over-time",
                            figure=topic_model.visualize_topics_over_time(
                                topics_over_time, 
                                top_n_topics=10
                            ),
                            style={"height": "600px"}
                        )
                    ], 
                    className="p-4"
                ),
            ])
        ], className="w-full lg:w-2/3 p-4")
    

    @staticmethod
    def create_topic_details_section(topic_docs, topic_model):
        """Create the topic details section with dropdown for topic selection"""
        if not topic_docs:
            return html.Div()
        
        # Create dropdown options from available topics
        topic_options = [
            {'label': f"Topic {topic_id}", 'value': topic_id}
            for topic_id in topic_docs.keys()
        ]
        
        # Define the function to create a topic card
        def create_topic_card(topic_id):
            if topic_id is None:
                return html.Div("Select a topic to view details", className="p-4 text-gray-500")
            
            docs = topic_docs.get(topic_id, [])
            topic_words = topic_model.get_topic(topic_id)
            topic_name = f"Topic {topic_id}"
            
            return html.Div([
                html.H3(topic_name, className="text-lg font-semibold mb-2"),
                html.Div([
                    html.Span("Keywords: ", className="font-semibold"),
                    html.Span(", ".join([word for word, _ in topic_words[:7]]))
                ], className="mb-2 text-sm"),
                html.H4("Sample Documents:", className="font-medium mb-1"),
                html.Ul([
                    html.Li([
                        html.Div(doc[:200] + "..." if len(doc) > 200 else doc),
                        html.Div(f"Probability: {prob:.4f}", className="text-xs text-gray-500")
                    ], className="mb-2") 
                    for doc, prob in docs
                ], className="list-disc pl-5 text-sm")
            ], className="p-4 border rounded bg-white")
        
        # Set up the layout with dropdown and content area
        return html.Div([
            html.H2("Topic Details", className="text-xl font-bold mb-3"),
            html.Div([
                html.Label("Select Topic:", className="font-medium mr-2"),
                dcc.Dropdown(
                    id="topic-dropdown",
                    options=topic_options,
                    value=None,
                    clearable=False,
                    className="w-64"
                )
            ], className="mb-4"),
            html.Div(id="topic-card-container", className="w-full")
        ], className="w-full p-4")


class DashboardBuilder:
    def __init__(self, topic_model, topics, probs, docs, timestamps, reduced_embeddings,
                 coherence_scores, topic_diversity):
        self.topic_model = topic_model
        self.topics = topics
        self.probs = probs 
        self.docs = docs
        self.timestamps = timestamps
        self.reduced_embeddings = reduced_embeddings
        self.coherence_scores = coherence_scores
        self.topic_diversity = topic_diversity
        self.topic_info = topic_model.get_topic_info()
        
        # Calculate additional data for dashboard
        self.topic_docs = self._get_topic_documents()
        self.topics_over_time = topic_model.topics_over_time(docs, timestamps=timestamps)
        
    def _get_topic_documents(self):
        """Get sample documents for each topic"""
        topic_docs = {}
        for topic_id in set(self.topics):
            if topic_id == -1:  # Skip outliers
                continue
                
            indices = [i for i, t in enumerate(self.topics) if t == topic_id]
            if indices:
                # Get up to 5 sample documents for each topic
                sample_indices = indices[:5]
                topic_docs[topic_id] = [(self.docs[i], self.probs[i]) for i in sample_indices]
                
        return topic_docs
        
    def create_layout(self):
        """Create the complete dashboard layout"""
        return html.Div([
            # Header
            html.Div([
                html.H1("Review Topics Dashboard", className="text-2xl font-bold mb-2"),
                html.P("Interactive visualization of topic modeling results", className="text-gray-600"),
            ], className="p-4 bg-gray-100 border-b"),
            
            # Main content area
            html.Div([
                # Left column: Model performance metrics
                DashboardComponents.create_model_performance_section(
                    self.coherence_scores, 
                    self.topic_diversity,
                    self.topic_info
                ),
                
                # Right column: Topic visualizations
                DashboardComponents.create_topic_visualizations(
                    self.topic_model,
                    self.topics,
                    self.reduced_embeddings,
                    self.docs,
                    self.topics_over_time
                ),
            ], className="flex flex-wrap"),
            
            # Bottom section: Topic details with sample documents
            DashboardComponents.create_topic_details_section(
                self.topic_docs,
                self.topic_model
            )
        ])
        
    def build_dashboard(self, port=8050, debug=False, dev_tools_hot_reload=False):
        """Create and run the dashboard application"""
        app = dash.Dash(
            __name__,
            external_stylesheets=[
                "https://cdnjs.cloudflare.com/ajax/libs/tailwindcss/2.2.19/tailwind.min.css"
            ]
        )
        
        app.layout = self.create_layout()
        
        # Add callbacks
        self._add_callbacks(app)
        
        # Return the app
        return app
        
    def _add_callbacks(self, app):
        """Add interactive callbacks to the dashboard"""
        
        # Callback to update the topic card when dropdown selection changes
        @app.callback(
            Output("topic-card-container", "children"),
            Input("topic-dropdown", "value")
        )
        def update_topic_card(selected_topic):
            if selected_topic is None:
                return html.Div("Select a topic to view details", className="p-4 text-gray-500")
            
            # Get topic details
            docs = self.topic_docs.get(selected_topic, [])
            topic_words = self.topic_model.get_topic(selected_topic)
            topic_name = f"Topic {selected_topic}"
            
            # Create card for the selected topic
            return html.Div([
                html.H3(topic_name, className="text-lg font-semibold mb-2"),
                html.Div([
                    html.Span("Keywords: ", className="font-semibold"),
                    html.Span(", ".join([word for word, _ in topic_words[:7]]))
                ], className="mb-2 text-sm"),
                html.H4("Sample Documents:", className="font-medium mb-1"),
                html.Ul([
                    html.Li([
                        html.Div(doc[:200] + "..." if len(doc) > 200 else doc),
                        html.Div(f"Probability: {prob:.4f}", className="text-xs text-gray-500")
                    ], className="mb-2") 
                    for doc, prob in docs
                ], className="list-disc pl-5 text-sm")
            ], className="p-4 border rounded bg-white")

# Main application 
def main():
    # Load configuration
    parser = argparse.ArgumentParser(description="Topic Modeling with BERTopic")
    parser.add_argument("--config", default="config.yaml", help="Path to configuration file")
    args = parser.parse_args()
    
    try:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
            logger.info("Configuration loaded successfully")
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        raise
    
    try:
        # Load data
        data_loader = DataLoader(config["data"])
        reviews_df = data_loader.load_reviews()
        documents = reviews_df["review"].tolist()
        timestamps = reviews_df["timestamp"].tolist()
        
        # Configure and train model
        bert_config = BERTopicConfigurator(config["model"])
        modeler = TopicModeler(bert_config.topic_model)
        topics, probs, embeddings, reduced_embeddings = modeler.fit(documents)
        
        # Evaluate model
        evaluator = Evaluator(modeler.topic_model, config["evaluation"])
        metrics = evaluator.calculate_metrics(documents)
        
        # Save model
        if config.get("save_model", False):
            output_dir = config.get("output_dir", "./results")
            os.makedirs(output_dir, exist_ok=True)
            modeler.save(output_dir)
        
        # Build and run dashboard
        dashboard_builder = DashboardBuilder(
            topic_model=modeler.topic_model,
            topics=topics,
            probs=probs,
            docs=documents,
            timestamps=timestamps,
            reduced_embeddings=reduced_embeddings,
            coherence_scores=metrics["coherence_scores"],
            topic_diversity=metrics["topic_diversity"]
        )
        
        app = dashboard_builder.build_dashboard(
            port=config.get("dashboard", {}).get("port", 8050),
            debug=config.get("dashboard", {}).get("debug", True)
        )
        
        # Run the dashboard
        app.run(
            host=config.get("dashboard", {}).get("host", "0.0.0.0"),
            port=config.get("dashboard", {}).get("port", 8050),
            debug=config.get("dashboard", {}).get("debug", True)
        )
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}\n{traceback.format_exc()}")
        raise

if __name__ == "__main__":
    # Set environment variable to avoid tokenizers parallelism issues
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()