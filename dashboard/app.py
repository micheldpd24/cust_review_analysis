
import dash
from dash import dcc, html, Input, Output, dash_table, State
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import base64
import io
from collections import Counter
from collections import Counter
import nltk
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
import os

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')


# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)

# Load data
DATA_PATH = "/data/full_reviews.csv"
try:
    data = pd.read_csv(DATA_PATH)
    # Ensure required columns are present
    required_columns = ["year", "yearQuarter", "yearMonth", "date", "sentiment", "rating", "review", "reply", "id"]
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing columns in data: {missing_columns}")
except FileNotFoundError:
    data = pd.DataFrame(columns=["year", "yearQuarter", "yearMonth", "date", "sentiment", "rating", "review", "reply", "id"])
    print("Warning: Data file not found. Using empty DataFrame.")
except ValueError as e:
    print(f"Data error: {e}")
    data = pd.DataFrame(columns=["year", "yearQuarter", "yearMonth", "date", "sentiment", "rating", "review", "reply", "id"])


# Reusable Graph Component
def create_graph(id, figure=None):
    return dcc.Graph(id=id, figure=figure if figure else px.line(title="Loading..."), 
                     style={"height": "400px"})

# Global Filters with Clarity
filter_section = dbc.Row([
    html.Div("Filters (Apply to Overview and Textual Analysis only):", style={"marginBottom": 10}),
    dbc.Col(dcc.Dropdown(id="year-filter", options=[{"label": str(y), "value": y} for y in data["year"].unique()], 
                         multi=True, placeholder="Select Year(s)"), width={"size": 3, "offset": 0}),
    dbc.Col(dcc.Dropdown(id="quarter-filter", options=[{"label": q, "value": q} for q in data["yearQuarter"].unique()], 
                         multi=True, placeholder="Select Quarter(s)"), width={"size": 3, "offset": 0}),
    dbc.Col(dcc.Dropdown(id="month-filter", options=[{"label": m, "value": m} for m in data["yearMonth"].unique()], 
                         multi=True, placeholder="Select Month(s)"), width={"size": 3, "offset": 0}),
    dbc.Col(html.Button("Reset Filters", id="reset-button", className="btn btn-secondary"), width={"size": 3, "offset": 0}),
], style={"marginBottom": 20, "marginTop": 20})

# App Layout
app.layout = dbc.Container([
    html.H1("Back Market Customer Reviews Dashboard", style={"textAlign": "center", "marginBottom": 20}),
    dcc.Interval(id="interval-component", interval=50000, n_intervals=0),  # Check every 5 seconds
    dcc.Store(id="file-timestamp", data=os.path.getmtime(DATA_PATH) if os.path.exists(DATA_PATH) else None),
    filter_section,
    dcc.Tabs(id="tabs", value="tab-1", children=[
        dcc.Tab(label="Overall Overview", value="tab-1"),
        dcc.Tab(label="Textual Analysis", value="tab-3"),
        dcc.Tab(label="Trends", value="tab-2"),
    ]),
    dcc.Loading(id="loading", children=html.Div(id="tab-content")),
], fluid=True)

@app.callback(
    [Output("file-timestamp", "data"), Output("tabs", "value")],  # Stocke le nouvel horodatage et rafraîchit l'onglet actif
    [Input("interval-component", "n_intervals")],  # Vérifie périodiquement
    [State("file-timestamp", "data")]
)
def check_file_update(n, last_timestamp):
    """ Vérifie si le fichier CSV a été modifié et met à jour les données si nécessaire. """
    if os.path.exists(DATA_PATH):
        new_timestamp = os.path.getmtime(DATA_PATH)
        if last_timestamp is None or new_timestamp > last_timestamp:
            print("File updated, reloading data...")
            global data
            data = pd.read_csv(DATA_PATH)  # Recharge les données
            return new_timestamp, "tab-1"  # Met à jour le timestamp et rafraîchit l'onglet actif
    return last_timestamp, dash.no_update  # Ne rien faire si le fichier n'a pas changé

# Filter Data Helper Function (Optimized)

def filter_data(years, quarters, months):
    mask = pd.Series(True, index=data.index)
    if years:
        mask &= data["year"].isin(years)
    if quarters:
        mask &= data["yearQuarter"].isin(quarters)
    if months:
        mask &= data["yearMonth"].isin(months)
    filtered_data = data[mask]
    return filtered_data if not filtered_data.empty else None

# Tab Content Callback
@app.callback(
    Output("tab-content", "children"),
    [Input("tabs", "value"), Input("year-filter", "value"), Input("quarter-filter", "value"), Input("month-filter", "value")]
)
def render_content(tab, years, quarters, months):
    filtered_data = filter_data(years, quarters, months)
    if tab == "tab-1":  # Overall Overview
        if filtered_data is None:
            return html.P("No data available for the selected filters.", style={"textAlign": "center", "color": "red", "marginTop": 20})
        return overall_overview_tab(filtered_data)
    elif tab == "tab-2":  # Trends (uses full data)
        return trends_tab()
    elif tab == "tab-3":  # Textual Analysis
        if filtered_data is None:
            return html.P("No data available for the selected filters.", style={"textAlign": "center", "color": "red", "marginTop": 20})
        return textual_analysis_tab(filtered_data)

# Overall Overview Tab
def overall_overview_tab(filtered_data):
    sentiment_pie = px.pie(filtered_data, names="sentiment", title="Sentiment Distribution", 
                           color_discrete_sequence=px.colors.qualitative.Pastel, hole=0.3)
    sentiment_pie.update_traces(textinfo="percent+label", hoverinfo="label+percent+value")
    
    rating_bar = px.histogram(filtered_data, x="rating", title="Rating Distribution", 
                              color="rating", text_auto=True, color_discrete_sequence=px.colors.sequential.Viridis)
    rating_bar.update_layout(bargap=0.2)
    
    reply_rate = filtered_data.groupby("year")["reply"].apply(lambda x: x.notnull().mean() * 100).reset_index()
    reply_rate_bar = px.bar(reply_rate, x="year", y="reply", title="Reply Rate by Year (%)", 
                            color="reply", color_continuous_scale="Blues")
    reply_rate_bar.update_traces(text=reply_rate["reply"].round(1), textposition="auto")
    
    avg_rating = go.Figure(go.Indicator(
        mode="gauge+number", value=filtered_data["rating"].mean(),
        title={"text": "Overall Average Rating"}, gauge={"axis": {"range": [0, 5]}, "bar": {"color": "darkblue"}}
    ))
    
    yearly_data = filtered_data.groupby("year").agg(
        review_volume=("id", "count"), average_rating=("rating", "mean")
    ).reset_index()
    table = dash_table.DataTable(
        data=yearly_data.to_dict("records"),
        columns=[
            {"name": "Year", "id": "year"},
            {"name": "Review Volume", "id": "review_volume"},
            {"name": "Avg Rating", "id": "average_rating", "type": "numeric", "format": {"specifier": ".2f"}}
        ],
        style_table={"overflowX": "auto"}, style_cell={"textAlign": "center"}, page_size=10,
        sort_action="native"
    )
    
    export_button = html.A("Export Data as CSV", id="export-button", download="filtered_reviews.csv", 
                           href="", className="btn btn-primary", style={"marginTop": 10})
    
    return dbc.Container([
        dbc.Row([
            dbc.Col(create_graph("sentiment-pie", sentiment_pie), width={"size": 6, "offset": 0}),
            dbc.Col(create_graph("rating-bar", rating_bar), width={"size": 6, "offset": 0})
        ]),
        dbc.Row([
            dbc.Col(create_graph("avg-rating", avg_rating), width={"size": 6, "offset": 0}),
            dbc.Col(create_graph("reply-rate-bar", reply_rate_bar), width={"size": 6, "offset": 0})
        ]),
        html.Div("Yearly Review Volume and Average Rating", style={"textAlign": "left", "marginBottom": 10, "marginTop": 10,}),
        dbc.Row(dbc.Col([table, export_button], width={"size": 12, "offset": 0}), style={"marginTop": 20}),
    ], fluid=True)

# Trends Tab
def trends_tab():
    trend_type_dropdown = dcc.Dropdown(id="trend-type", options=[
        {"label": "Yearly", "value": "year"}, {"label": "Quarterly", "value": "yearQuarter"}, 
        {"label": "Monthly", "value": "yearMonth"}, {"label": "Daily", "value": "date"}
    ], value="year", clearable=False)
    
    return dbc.Container([
        dbc.Row([dbc.Col(trend_type_dropdown, width={"size": 3, "offset": 0})], style={"marginBottom": 20}),
        dbc.Row(dbc.Col(create_graph("volume-trend"), width={"size": 12, "offset": 0})),
        dbc.Row(dbc.Col(create_graph("sentiment-trend"), width={"size": 12, "offset": 0})),
        dbc.Row(dbc.Col(create_graph("rating-trend"), width={"size": 12, "offset": 0})),
    ], fluid=True)

# Trends Tab Callback with Tooltips
@app.callback(
    [Output("volume-trend", "figure"), Output("sentiment-trend", "figure"), Output("rating-trend", "figure")],
    [Input("trend-type", "value")]
)
def update_trends(trend_type):
    if data.empty:
        empty_fig = px.line(title="No Data Available")
        return empty_fig, empty_fig, empty_fig
    
    trend_data = data.copy()
    volume_trend = trend_data.groupby(trend_type).size().reset_index(name="count")
    if volume_trend.empty:
        volume_fig = px.line(title=f"No Volume Data for {trend_type.capitalize()}")
    else:
        volume_fig = px.line(volume_trend, x=trend_type, y="count", title="Review Volume Trend", 
                             line_shape="spline", color_discrete_sequence=["#00CC96"], markers=True)
        volume_fig.update_traces(hovertemplate=f"{trend_type.capitalize()}: %{{x}}<br>Count: %{{y}}")
        volume_fig.update_layout(xaxis_title=trend_type.capitalize(), yaxis_title="Review Count")
    
    sentiment_trend = trend_data.groupby([trend_type, "sentiment"]).size().reset_index(name="count")
    if sentiment_trend.empty:
        sentiment_fig = px.line(title=f"No Sentiment Data for {trend_type.capitalize()}")
    else:
        sentiment_fig = px.line(sentiment_trend, x=trend_type, y="count", color="sentiment", 
                                title="Sentiment Trend", line_shape="spline", markers=True)
        sentiment_fig.update_traces(hovertemplate=f"{trend_type.capitalize()}: %{{x}}<br>Sentiment: %{{color}}<br>Count: %{{y}}")
        sentiment_fig.update_layout(xaxis_title=trend_type.capitalize(), yaxis_title="Count")
    
    rating_trend = trend_data.groupby(trend_type)["rating"].mean().reset_index()
    if rating_trend.empty:
        rating_fig = px.line(title=f"No Rating Data for {trend_type.capitalize()}")
    else:
        rating_fig = px.line(rating_trend, x=trend_type, y="rating", title="Average Rating Trend", 
                             line_shape="spline", color_discrete_sequence=["#EF553B"], markers=True)
        rating_fig.update_traces(hovertemplate=f"{trend_type.capitalize()}: %{{x}}<br>Avg Rating: %{{y:.2f}}")
        rating_fig.update_layout(xaxis_title=trend_type.capitalize(), yaxis_title="Average Rating")
    
    return volume_fig, sentiment_fig, rating_fig

# Textual Analysis Tab
def textual_analysis_tab(filtered_data):
    sentiment_dropdown = dcc.Dropdown(id="sentiment-filter", options=[
        {"label": "Positive", "value": "positive"}, {"label": "Neutral", "value": "neutral"}, 
        {"label": "Negative", "value": "negative"}
    ], value="positive", clearable=False)
    
    top_words_table = dash_table.DataTable(
        id="top-words-table",
        columns=[{"name": "N-gram", "id": "N-gram"}, {"name": "Count", "id": "Count"}],
        style_table={"overflowX": "auto"}, style_cell={"textAlign": "center"}, page_size=10
    )
    
    return dbc.Container([
        html.Div("Select a sentiment:", style={"textAlign": "left", "marginBottom": 10, "marginTop": 10,}),
        dbc.Row(dbc.Col(sentiment_dropdown, width={"size": 4, "offset": 0}), style={"marginBottom": 20}),
        html.Div("Word Cloud and  Top 10 n-grams for selected sentiment", style={"textAlign": "left", "marginBottom": 10, "marginTop": 10,}),
        html.Hr(style={'border': '1px solid blue', 'margin': '20px 0'}),
        dbc.Row([
            dbc.Col(html.Img(id="wordcloud-image", style={"width": "100%"}), width={"size": 8, "offset": 0}),
            dbc.Col(top_words_table, width={"size": 4, "offset": 0})
        ]),
    ], fluid=True)

# Textual Analysis Tab Callback
@app.callback(
    [Output("wordcloud-image", "src"), Output("top-words-table", "data")],
    [Input("sentiment-filter", "value"), Input("year-filter", "value"), 
     Input("quarter-filter", "value"), Input("month-filter", "value")]
)
def update_textual_analysis(sentiment, years, quarters, months):
    filtered_data = filter_data(years, quarters, months)
    if filtered_data is None or "review" not in filtered_data.columns or filtered_data["review"].isna().all():
        return "", []
    
    text_data = filtered_data[filtered_data["sentiment"] == sentiment]
    text = " ".join(text_data["review"].dropna().astype(str))
    
    if not text.strip():
        return "", []
    
    tokens = word_tokenize(text.lower())
    trigrams = list(ngrams(tokens, 3))
    trigram_strings = [" ".join(trigram) for trigram in trigrams]
    trigram_counts = Counter(trigram_strings).most_common(100)
    
    wc = WordCloud(width=800, height=400, background_color="white", max_words=100)
    wc.generate_from_frequencies(dict(trigram_counts))
    
    buf = io.BytesIO()
    wc.to_image().save(buf, format="PNG")
    image_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    wc_src = f"data:image/png;base64,{image_base64}"
    
    top_words = pd.DataFrame(trigram_counts, columns=["N-gram", "Count"])
    
    return wc_src, top_words.to_dict("records")


# Reset Filters Callback
@app.callback(
    [Output("year-filter", "value"), Output("quarter-filter", "value"), Output("month-filter", "value")],
    [Input("reset-button", "n_clicks")]
)
def reset_filters(n_clicks):
    return None, None, None

# Export Button Callback
@app.callback(
    Output("export-button", "href"),
    [Input("year-filter", "value"), Input("quarter-filter", "value"), Input("month-filter", "value")]
)
def update_export_link(years, quarters, months):
    filtered_data = filter_data(years, quarters, months)
    if filtered_data is None:
        return ""
    csv_string = filtered_data.to_csv(index=False)
    csv_base64 = base64.b64encode(csv_string.encode()).decode()
    return f"data:text/csv;base64,{csv_base64}"

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8050)