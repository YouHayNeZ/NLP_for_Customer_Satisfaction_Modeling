import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
from jupyter_dash import JupyterDash


def create_treemap_visualizations(comment_counts, topic_colors, sentiment_colors, note):
    """
    Create treemap visualizations for comments by topic and sentiment.

    Parameters:
    - comment_counts (pd.DataFrame): DataFrame containing comment counts with 'Topic', 'Sentiment', and 'Number of Comments' columns.
    - topic_colors (dict): Dictionary mapping topics to their respective colors.
    - sentiment_colors (dict): Dictionary mapping sentiments to their respective colors.
    - note (str): Annotation note to be added to the plots.

    Returns:
    - fig_topic (plotly.graph_objs._figure.Figure): Treemap figure with Topics, Sentiments.
    - fig_sen (plotly.graph_objs._figure.Figure): Treemap figure with Sentiments, Topics.
    """

    # Combine the two color mappings into one
    combined_colors = {**sentiment_colors, **topic_colors}

    # Create a combined key for color mapping
    comment_counts['color_key'] = comment_counts.apply(
        lambda row: row['Sentiment'] if row['Topic'] == '' else row['Topic'], axis=1)

    # Treemap by Topic and Sentiment
    fig_topic = px.treemap(comment_counts, path=['Topic', 'Sentiment'], values='Number of Comments',
                           color='color_key', color_discrete_map=combined_colors,
                           title='Treemap of Comments by Topic and Sentiment')

    fig_topic.add_annotation(
        text=note,
        xref='paper', yref='paper',
        x=0.5, y=-0.1,
        showarrow=False,
        font=dict(size=12),
        align='center'
    )

    # Treemap by Sentiment and Topic
    fig_sen = px.treemap(comment_counts, path=['Sentiment', 'Topic'], values='Number of Comments',
                         color='Sentiment', color_discrete_map=sentiment_colors,
                         title='Treemap of Comments by Sentiment and Topic')

    fig_sen.add_annotation(
        text=note,
        xref='paper', yref='paper',
        x=0.5, y=-0.1,
        showarrow=False,
        font=dict(size=12),
        align='center'
    )
    return fig_topic, fig_sen


def create_heatmap_visualization(long_df, note):
    """
    Create a heatmap visualization for comments by sentiment and topic.

    Parameters:
    - long_df (pd.DataFrame): DataFrame containing long-form data with 'Topic', 'Sentiment', and 'count' columns.
    - note (str): Annotation note to be added to the plot.

    Returns:
    - fig (plotly.graph_objs._figure.Figure): Heatmap figure.
    """

    # Aggregate the data to get the count of comments per combination
    heatmap_data = long_df.groupby(['Topic', 'Sentiment']).size().reset_index(name='count')

    # Create the heatmap plot
    fig = px.density_heatmap(heatmap_data, x='Sentiment', y='Topic', z='count',
                             title='Heatmap of Sentiment by Topic',
                             labels={'count': 'Number of Comments'},
                             color_continuous_scale='Blues')
    fig.update_layout(
        xaxis_title='Sentiment',
        yaxis_title='Topic',
        coloraxis_colorbar=dict(title='Number of Comments'),
        annotations=[{
            'x': 0.5, 'y': -0.2, 'xref': 'paper', 'yref': 'paper',
            'text': note,
            'showarrow': False,
            'font': {'size': 12}
        }]
    )

    return fig


def create_dash_app_sentiments_over_time(df, topics, sentiment_colors, note, category_orders):
    """
    Create a Dash app for visualizing sentiment trends over time.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the data.
    - topics (dict): Dictionary mapping topics to their respective columns.
    - sentiment_colors (dict): Dictionary mapping sentiments to their respective colors.
    - note (str): Annotation note to be added to the plots.
    - category_orders (dict): Dictionary defining the order of categories.

    Returns:
    - app (JupyterDash): The Dash app object.
    """

    app = JupyterDash('Sentiment Distribution Per Topic Over Time')

    app.layout = html.Div([
        html.H1("Sentiment Trends Over Time", style={'textAlign': 'center'}),
        html.Div([
            html.Label('Topic:'),
            dcc.Dropdown(
                id='topic-dropdown',
                options=[{'label': label, 'value': value} for label, value in topics.items()],
                multi=True
            ),
            dcc.Graph(id='sentiment-trends')
        ], style={'width': '100%', 'margin': '0 auto'})
    ])

    @app.callback(
        Output('sentiment-trends', 'figure'),
        [Input('topic-dropdown', 'value')]
    )
    def update_graph(selected_topics):
        filtered_df = df.copy()

        # Filter by selected topics
        if selected_topics:
            for topic in selected_topics:
                filtered_df = filtered_df[filtered_df[topic] == True]

        filtered_df['Year'] = filtered_df['Date Published Formatted'].dt.year
        yearly_data = filtered_df.groupby(['Year', 'Sentiment']).size().reset_index(name='count')

        total_yearly_comments = filtered_df.groupby('Year').size().reset_index(name='total')
        yearly_data = yearly_data.merge(total_yearly_comments, on='Year')
        yearly_data['percentage'] = yearly_data['count'] / yearly_data['total'] * 100

        fig = px.line(yearly_data, x='Year', y='percentage', color='Sentiment',
                      title='Sentiment Trends Over Time',
                      color_discrete_map=sentiment_colors,
                      labels={'percentage': 'Percentage of Comments', 'Year': 'Year'},
                      category_orders=category_orders)

        fig.update_layout(
            xaxis_title='Year',
            yaxis_title='Percentage of Comments',
            legend_title='Sentiment',
            hovermode='x unified',
            annotations=[{
                'x': 0.5, 'y': -0.3, 'xref': 'paper', 'yref': 'paper',
                'text': note,
                'showarrow': False,
                'font': {'size': 12}
            }]
        )

        return fig

    return app

def create_dash_app_topics_over_time(df, topics, note):
    """
    Create a Dash app for visualizing the total number of comments per topic over time.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the data.
    - topics (dict): Dictionary mapping topics to their respective columns.
    - note (str): Annotation note to be added to the plots.

    Returns:
    - app (JupyterDash): The Dash app object.
    """

    app = JupyterDash("Total Number of Comments per Topic Over Time")

    app.layout = html.Div([
        html.H1("Total Number of Comments per Topic Over Time", style={'textAlign': 'center'}),
        html.Div([
            html.Label('Topic:'),
            dcc.Dropdown(
                id='topic-dropdown',
                options=[{'label': label, 'value': value} for label, value in topics.items()],
                multi=True
            ),
            dcc.Graph(id='topic-trends')
        ], style={'width': '100%', 'margin': '0 auto'})
    ])

    @app.callback(
        Output('topic-trends', 'figure'),
        [Input('topic-dropdown', 'value')]
    )
    def update_graph(selected_topics):
        filtered_df = df.copy()

        # Filter by selected topics
        if selected_topics:
            for topic in selected_topics:
                filtered_df = filtered_df[filtered_df[topic] == True]

        # Convert date to year
        filtered_df['Year'] = filtered_df['Date Published Formatted'].dt.year

        # Initialize an empty DataFrame for yearly data
        yearly_data = pd.DataFrame()

        # Aggregate data for each topic
        for topic_name, topic_col in topics.items():
            topic_df = filtered_df[filtered_df[topic_col] == True]
            topic_yearly = topic_df.groupby(['Year']).size().reset_index(name='count')
            topic_yearly['Topic'] = topic_name
            yearly_data = pd.concat([yearly_data, topic_yearly])

        # Create the line plot
        fig = px.line(yearly_data, x='Year', y='count', color='Topic',
                      title='Total Number of Comments per Topic Over Time',
                      labels={'count': 'Number of Comments', 'Year': 'Year'})

        # Update layout for better readability
        fig.update_layout(
            xaxis_title='Year',
            yaxis_title='Number of Comments',
            legend_title='Topic',
            hovermode='x unified',
            annotations=[{
                'x': 0.5, 'y': -0.3, 'xref': 'paper', 'yref': 'paper',
                'text': note,
                'showarrow': False,
                'font': {'size': 12}
            }]
        )

        return fig

    return app


def create_sentiment_distribution_app(df, categories, topics, sentiment_colors):
    """
    Create a Dash app for visualizing the distribution of comments per sentiment.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the data.
    - categories (list): List of categories for filters.
    - topics (dict): Dictionary mapping topics to their respective columns.
    - sentiment_colors (dict): Dictionary mapping sentiments to their respective colors.

    Returns:
    - app (JupyterDash): The Dash app object.
    """

    app = JupyterDash("Distribution of Number Comments Per Sentiment")

    # Layout
    app.layout = html.Div([
        html.H1("Distribution of Comment Per Sentiment", style={'textAlign': 'center'}),
        html.Div([
            html.Div([
                html.Div([
                    html.Label(f'{category}:'),
                    dcc.Dropdown(
                        id=f'{category}-dropdown',
                        options=[{'label': value, 'value': value} for value in df[category].dropna().unique()],
                        multi=True
                    )
                ]) for category in categories
            ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '20px'}),
            html.Div([
                html.Label('Topic:'),
                dcc.Dropdown(
                    id='topic-dropdown',
                    options=[{'label': label, 'value': value} for label, value in topics.items()],
                    multi=True
                ),
                dcc.Graph(id='sentiment-distribution')
            ], style={'width': '70%', 'display': 'inline-block', 'verticalAlign': 'top'})
        ], style={'display': 'flex'})
    ])

    @app.callback(
        Output('sentiment-distribution', 'figure'),
        [Input(f'{category}-dropdown', 'value') for category in categories] +
        [Input('topic-dropdown', 'value')]
    )
    def update_graph(*filter_values):
        filtered_df = df.copy()
        topic_filter_values = filter_values[-1]
        category_filter_values = filter_values[:-1]

        # Filter by selected categories
        for category, filter_value in zip(categories, category_filter_values):
            if filter_value:
                filtered_df = filtered_df[filtered_df[category].isin(filter_value)]

        # Filter by selected topics
        if topic_filter_values:
            for topic in topic_filter_values:
                filtered_df = filtered_df[filtered_df[topic] == True]

        comment_count = len(filtered_df)
        fig = px.pie(filtered_df, names='Sentiment', title=f'Total Number of Comments: {comment_count}',
                     color='Sentiment', color_discrete_map=sentiment_colors)
        fig.update_traces(textinfo='label+percent+value', insidetextorientation='radial')

        return fig

    return app


def create_sunburst_plot(long_df, note):
    """
    Create a Sunburst plot for sentiment breakdown by topic.

    Parameters:
    - long_df (pd.DataFrame): DataFrame containing long-form data with 'Topic', 'Sentiment', and 'count' columns.
    - note (str): Annotation note to be added to the plot.

    Returns:
    - fig (plotly.graph_objs._figure.Figure): Sunburst plot figure.
    """

    sunburst_data = long_df.groupby(['Topic', 'Sentiment']).size().reset_index(name='count')

    fig = px.sunburst(sunburst_data, path=['Topic', 'Sentiment'], values='count',
                      title='Sentiment Breakdown by Topic')

    fig.add_annotation(
        text=note,
        xref='paper', yref='paper',
        x=0.5, y=-0.1,
        showarrow=False,
        font=dict(size=12),
        align='center'
    )

    return fig
