import pandas as pd
import plotly.express as px
from dash import dcc, html
from jupyter_dash import JupyterDash
from dash.dependencies import Input, Output

"""
This file stores the functions for the jupyter notebook: dashboard.ipynb
Please only invoke the functions from the jupyter notebook.
"""


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

    combined_colors = {**sentiment_colors, **topic_colors}

    comment_counts['color_key'] = comment_counts.apply(
        lambda row: row['Sentiment'] if row['Topic'] == '' else row['Topic'], axis=1)

    # Treemap by Topic and Sentiment
    fig_topic = px.treemap(comment_counts, path=['Topic', 'Sentiment'], values='Number of Comments',
                           color='color_key', color_discrete_map=combined_colors,
                           title='Number of Comments* Breakdown by Topic and Sentiment',
                           custom_data=['Number of Comments'])

    fig_topic.update_traces(
        hovertemplate='<b>%{label}</b><br>Number of Comments: %{customdata[0]}<extra></extra>',
        texttemplate='%{label}<br>%{customdata[0]}',
        textfont=dict(size=12)
    )

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
                         title='Number of Comments* Breakdown by Sentiment and Topic',
                         custom_data=['Number of Comments'])

    fig_sen.update_traces(
        hovertemplate='<b>%{label}</b><br>Number of Comments: %{customdata[0]}<extra></extra>',
        texttemplate='%{label}<br>%{customdata[0]}',
        textfont=dict(size=12)
    )

    fig_sen.add_annotation(
        text=note,
        xref='paper', yref='paper',
        x=0.5, y=-0.1,
        showarrow=False,
        font=dict(size=12),
        align='center'
    )

    return fig_topic, fig_sen


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
                      title='Number of Comments* Breakdown by Topic and Sentiment')

    fig.add_annotation(
        text=note,
        xref='paper', yref='paper',
        x=0.5, y=-0.1,
        showarrow=False,
        font=dict(size=12),
        align='center'
    )

    return fig


def create_heatmap_visualization(long_df, note):
    """
    Create a heatmap visualization for comments by sentiment and topic.

    Parameters:
    - long_df (pd.DataFrame): DataFrame containing long-form data with 'Topic', 'Sentiment', and 'count' columns.
    - note (str): Annotation note to be added to the plot.

    Returns:
    - fig (plotly.graph_objs._figure.Figure): Heatmap figure.
    """

    heatmap_data = long_df.groupby(['Topic', 'Sentiment']).size().reset_index(name='count')

    fig = px.density_heatmap(heatmap_data, x='Sentiment', y='Topic', z='count',
                             title='Number of Comments* Breakdown by Topic and Sentiment',
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


def create_dash_app_sentiments_over_time(df, topics, sentiment_colors, category_orders):
    """
    Create a Dash app_sentiment_plot for visualizing sentiment trends over time.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the data.
    - topics (dict): Dictionary mapping topics to their respective columns.
    - sentiment_colors (dict): Dictionary mapping sentiments to their respective colors.
    - note (str): Annotation note to be added to the plots.
    - category_orders (dict): Dictionary defining the order of categories.

    Returns:
    - app_sentiment_plot (JupyterDash): The Dash app_sentiment_plot object.
    """

    app_name = "Sentiment Distribution Over Time with Topic Filter"

    app_sentiment_plot = JupyterDash(app_name)

    app_sentiment_plot.layout = html.Div([
        html.H1(app_name, style={'textAlign': 'center'}),
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

    @app_sentiment_plot.callback(
        Output('sentiment-trends', 'figure'),
        [Input('topic-dropdown', 'value')]
    )
    def update_graph(topic_filter_values):
        filtered_df = df.copy()

        if topic_filter_values:
            for topic in topic_filter_values:
                filtered_df = filtered_df[filtered_df[topic] == True]

        filtered_df['Year'] = filtered_df['Date Published Formatted'].dt.year
        yearly_data = filtered_df.groupby(['Year', 'Sentiment']).size().reset_index(name='count')

        total_yearly_comments = filtered_df.groupby('Year').size().reset_index(name='total')
        yearly_data = yearly_data.merge(total_yearly_comments, on='Year')
        yearly_data['percentage'] = yearly_data['count'] / yearly_data['total'] * 100

        fig = px.line(yearly_data, x='Year', y='percentage', color='Sentiment',
                      title='Sentiment Distribution Over Time',
                      color_discrete_map=sentiment_colors,
                      labels={'percentage': 'Percentage of Comments', 'Year': 'Year'},
                      category_orders=category_orders)

        fig.update_layout(
            xaxis_title='Year',
            yaxis_title='Percentage of Comments (%)',
            legend_title='Sentiment',
            hovermode='x unified'
        )

        return fig

    return app_sentiment_plot


def create_dash_app_topics_over_time(df, topics, category_orders):
    """
    Create a Dash app_topic_plot for visualizing the total number of comments per topic over time.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the data.
    - topics (dict): Dictionary mapping topics to their respective columns.
    - note (str): Annotation note to be added to the plots.

    Returns:
    - app_topic_plot (JupyterDash): The Dash app_topic_plot object.
    """

    app_name="Topic Distribution over Time with Topic and Sentiment Filters"

    app_topic_plot = JupyterDash(app_name)

    app_topic_plot.layout = html.Div([
        html.H1(app_name, style={'textAlign': 'center'}),
        html.Div([
            html.Label('Topic:'),
            dcc.Dropdown(
                id='topic-dropdown',
                options=[{'label': label, 'value': value} for label, value in topics.items()],
                multi=True
            ),
            html.Label('Sentiment:'),
            dcc.Dropdown(
                id='sentiment-dropdown',
                options=[{'label': sentiment, 'value': sentiment} for sentiment in df['Sentiment'].unique()],
                multi=True
            ),
            dcc.Graph(id='topic-trends')
        ], style={'width': '100%', 'margin': '0 auto'})
    ])

    @app_topic_plot.callback(
        Output('topic-trends', 'figure'),
        [Input('topic-dropdown', 'value'),
         Input('sentiment-dropdown', 'value')]
    )
    def update_graph(selected_topics, selected_sentiments):
        filtered_df = df.copy()
        if selected_topics:
            for topic in selected_topics:
                filtered_df = filtered_df[filtered_df[topic] == True]
        if selected_sentiments:
            for sentiment in selected_sentiments:
                filtered_df = filtered_df[filtered_df['Sentiment'] == sentiment]

        filtered_df['Year'] = filtered_df['Date Published Formatted'].dt.year

        yearly_data = pd.DataFrame()

        for topic_name, topic_col in topics.items():
            topic_df = filtered_df[filtered_df[topic_col] == True]
            topic_yearly = topic_df.groupby(['Year']).size().reset_index(name='count')
            topic_yearly['Topic'] = topic_name
            yearly_data = pd.concat([yearly_data, topic_yearly])

        fig = px.line(yearly_data, x='Year', y='count', color='Topic',
                      title='Topic Distribution over Time',
                      labels={'count': 'Number of Comments', 'Year': 'Year'})

        fig.update_layout(
            xaxis_title='Year',
            yaxis_title='Number of Comments',
            legend_title='Topic',
            hovermode='x unified',
        )

        return fig

    return app_topic_plot


def create_sentiment_distribution_app(df, categories, topics, sentiment_colors):
    """
    Create a Dash app_sentiment_pie for visualizing the distribution of comments per sentiment.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the data.
    - categories (list): List of categories for filters.
    - topics (dict): Dictionary mapping topics to their respective columns.
    - sentiment_colors (dict): Dictionary mapping sentiments to their respective colors.

    Returns:
    - app_sentiment_pie (JupyterDash): The Dash app_sentiment_pie object.
    """

    app_name =  "Distribution of Number Comments Per Sentiment with Filters"

    app_sentiment_pie = JupyterDash(app_name)

    if 'Date Flown' in categories:
        df['Date Flown Parsed'] = pd.to_datetime(df['Date Flown'], format='%B %Y', errors='coerce')
        sorted_dates = df.dropna(subset=['Date Flown Parsed']).sort_values('Date Flown Parsed')['Date Flown'].unique()
        df.drop(columns=['Date Flown Parsed'], inplace=True)

    app_sentiment_pie.layout = html.Div([
        html.H1(app_name, style={'textAlign': 'center'}),
        html.Div([
            html.Div([
                html.Div([
                    html.Label(f'{category}:'),
                    dcc.Dropdown(
                        id=f'{category}-dropdown',
                        options=[{'label': value, 'value': value} for value in (
                            sorted_dates if category == 'Date Flown' else sorted(df[category].dropna().unique()))],
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

    @app_sentiment_pie.callback(
        Output('sentiment-distribution', 'figure'),
        [Input(f'{category}-dropdown', 'value') for category in categories] +
        [Input('topic-dropdown', 'value')]
    )
    def update_graph(*filter_values):
        filtered_df = df.copy()
        topic_filter_values = filter_values[-1]
        category_filter_values = filter_values[:-1]

        for category, filter_value in zip(categories, category_filter_values):
            if filter_value:
                filtered_df = filtered_df[filtered_df[category].isin(filter_value)]

        if topic_filter_values:
            for topic in topic_filter_values:
                filtered_df = filtered_df[filtered_df[topic] == True]

        comment_count = len(filtered_df)
        fig = px.pie(filtered_df, names='Sentiment', title=f'Total Number of Comments: {comment_count}',
                     color='Sentiment', color_discrete_map=sentiment_colors)
        fig.update_traces(textinfo='label+percent+value', insidetextorientation='radial')

        return fig

    return app_sentiment_pie
