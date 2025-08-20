#!/usr/bin/env python3
"""
Results Analysis Dashboard for NeuroBM.

This dashboard provides comprehensive analysis and comparison of experimental results
from multiple training runs and model configurations.

Features:
- Multi-experiment comparison
- Statistical analysis of results
- Performance metrics visualization
- Hyperparameter impact analysis
- Export and reporting capabilities

Usage:
    python dashboards/results_analyzer.py --results_dir=results/
    python dashboards/results_analyzer.py --port=8052
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import json
import glob
import pickle

import dash
from dash import dcc, html, Input, Output, State, callback, dash_table
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
from scipy import stats

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
app = dash.Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])
results_data = {}
experiments_df = pd.DataFrame()


def load_results_from_directory(results_dir: str) -> Dict[str, Any]:
    """Load all experimental results from directory."""
    results = {}
    results_path = Path(results_dir)
    
    if not results_path.exists():
        logger.warning(f"Results directory does not exist: {results_dir}")
        return results
    
    # Look for experiment directories
    for exp_dir in results_path.iterdir():
        if exp_dir.is_dir():
            exp_name = exp_dir.name
            exp_results = {}
            
            # Load training history
            history_file = exp_dir / 'training_history.json'
            if history_file.exists():
                try:
                    with open(history_file, 'r') as f:
                        exp_results['training_history'] = json.load(f)
                except Exception as e:
                    logger.warning(f"Could not load training history for {exp_name}: {e}")
            
            # Load evaluation results
            eval_file = exp_dir / 'evaluation_report.json'
            if eval_file.exists():
                try:
                    with open(eval_file, 'r') as f:
                        exp_results['evaluation'] = json.load(f)
                except Exception as e:
                    logger.warning(f"Could not load evaluation for {exp_name}: {e}")
            
            # Load config
            config_file = exp_dir / 'config.yaml'
            if config_file.exists():
                try:
                    import yaml
                    with open(config_file, 'r') as f:
                        exp_results['config'] = yaml.safe_load(f)
                except Exception as e:
                    logger.warning(f"Could not load config for {exp_name}: {e}")
            
            if exp_results:
                results[exp_name] = exp_results
                logger.info(f"Loaded results for experiment: {exp_name}")
    
    return results


def create_experiments_dataframe(results: Dict[str, Any]) -> pd.DataFrame:
    """Create DataFrame with experiment summary."""
    data = []
    
    for exp_name, exp_data in results.items():
        row = {'experiment': exp_name}
        
        # Extract config information
        if 'config' in exp_data:
            config = exp_data['config']
            row.update({
                'model_type': config.get('model', {}).get('type', 'unknown'),
                'hidden_units': config.get('model', {}).get('architecture', {}).get('n_hidden', 'unknown'),
                'learning_rate': config.get('training', {}).get('learning_rate', 'unknown'),
                'epochs': config.get('training', {}).get('epochs', 'unknown'),
                'batch_size': config.get('training', {}).get('batch_size', 'unknown')
            })
        
        # Extract final training metrics
        if 'training_history' in exp_data:
            history = exp_data['training_history']
            if 'train_loss' in history and history['train_loss']:
                row['final_train_loss'] = history['train_loss'][-1]
            if 'val_loss' in history and history['val_loss']:
                row['final_val_loss'] = history['val_loss'][-1]
        
        # Extract evaluation metrics
        if 'evaluation' in exp_data:
            eval_data = exp_data['evaluation']
            if 'reconstruction' in eval_data:
                recon = eval_data['reconstruction']
                row.update({
                    'mse': recon.get('mse', 'N/A'),
                    'mae': recon.get('mae', 'N/A'),
                    'correlation': recon.get('correlation', 'N/A')
                })
            
            if 'likelihood' in eval_data:
                likelihood = eval_data['likelihood']
                row['log_likelihood'] = likelihood.get('log_likelihood', 'N/A')
        
        data.append(row)
    
    return pd.DataFrame(data)


def create_layout():
    """Create the dashboard layout."""
    return html.Div([
        # Header
        html.Div([
            html.H1("📊 NeuroBM Results Analyzer", 
                   style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': 20}),
            html.P("Comprehensive analysis and comparison of experimental results",
                   style={'textAlign': 'center', 'color': '#7f8c8d', 'fontSize': 16})
        ], style={'backgroundColor': '#ecf0f1', 'padding': '20px', 'marginBottom': '20px'}),
        
        # Data Loading Panel
        html.Div([
            html.H3("📁 Data Loading"),
            html.Div([
                html.Div([
                    html.Label("Results Directory:"),
                    dcc.Input(
                        id='results-dir-input',
                        type='text',
                        placeholder='results/',
                        value='results/',
                        style={'width': '100%'}
                    )
                ], className='eight columns'),
                
                html.Div([
                    html.Button('🔄 Load Results', id='load-results-btn', 
                               className='button-primary', style={'marginTop': '25px'})
                ], className='four columns')
            ], className='row'),
            
            html.Div(id='load-status', style={'marginTop': '10px'})
        ], style={'backgroundColor': '#f8f9fa', 'padding': '20px', 'marginBottom': '20px'}),
        
        # Experiments Overview
        html.Div([
            html.H3("🔍 Experiments Overview"),
            html.Div(id='experiments-table-container')
        ], style={'marginBottom': '20px'}),
        
        # Analysis Tabs
        html.Div([
            dcc.Tabs(id='analysis-tabs', value='comparison', children=[
                dcc.Tab(label='📈 Performance Comparison', value='comparison'),
                dcc.Tab(label='📊 Training Curves', value='training'),
                dcc.Tab(label='🎯 Hyperparameter Analysis', value='hyperparams'),
                dcc.Tab(label='📋 Statistical Summary', value='statistics')
            ]),
            
            html.Div(id='analysis-content')
        ]),
        
        # Store for results data
        dcc.Store(id='results-data-store')
    ])


def create_comparison_tab():
    """Create performance comparison tab."""
    return html.Div([
        html.H4("Performance Comparison"),
        html.Div([
            html.Div([
                html.Label("Metric to Compare:"),
                dcc.Dropdown(
                    id='comparison-metric-dropdown',
                    options=[
                        {'label': 'Final Training Loss', 'value': 'final_train_loss'},
                        {'label': 'Final Validation Loss', 'value': 'final_val_loss'},
                        {'label': 'MSE', 'value': 'mse'},
                        {'label': 'MAE', 'value': 'mae'},
                        {'label': 'Correlation', 'value': 'correlation'},
                        {'label': 'Log Likelihood', 'value': 'log_likelihood'}
                    ],
                    value='final_train_loss'
                )
            ], className='six columns'),
            
            html.Div([
                html.Label("Chart Type:"),
                dcc.Dropdown(
                    id='chart-type-dropdown',
                    options=[
                        {'label': 'Bar Chart', 'value': 'bar'},
                        {'label': 'Box Plot', 'value': 'box'},
                        {'label': 'Scatter Plot', 'value': 'scatter'}
                    ],
                    value='bar'
                )
            ], className='six columns')
        ], className='row'),
        
        html.Div([
            dcc.Graph(id='comparison-plot')
        ], style={'marginTop': '20px'})
    ])


def create_training_tab():
    """Create training curves tab."""
    return html.Div([
        html.H4("Training Curves Analysis"),
        html.Div([
            html.Div([
                html.Label("Select Experiments:"),
                dcc.Dropdown(
                    id='training-experiments-dropdown',
                    multi=True,
                    placeholder="Select experiments to compare"
                )
            ], className='twelve columns')
        ], className='row'),
        
        html.Div([
            html.Div([
                dcc.Graph(id='training-loss-plot')
            ], className='six columns'),
            
            html.Div([
                dcc.Graph(id='validation-loss-plot')
            ], className='six columns')
        ], className='row', style={'marginTop': '20px'})
    ])


def create_hyperparams_tab():
    """Create hyperparameter analysis tab."""
    return html.Div([
        html.H4("Hyperparameter Impact Analysis"),
        html.Div([
            html.Div([
                html.Label("X-Axis Parameter:"),
                dcc.Dropdown(
                    id='x-param-dropdown',
                    options=[
                        {'label': 'Learning Rate', 'value': 'learning_rate'},
                        {'label': 'Hidden Units', 'value': 'hidden_units'},
                        {'label': 'Batch Size', 'value': 'batch_size'},
                        {'label': 'Epochs', 'value': 'epochs'}
                    ],
                    value='learning_rate'
                )
            ], className='six columns'),
            
            html.Div([
                html.Label("Y-Axis Metric:"),
                dcc.Dropdown(
                    id='y-metric-dropdown',
                    options=[
                        {'label': 'Final Training Loss', 'value': 'final_train_loss'},
                        {'label': 'Final Validation Loss', 'value': 'final_val_loss'},
                        {'label': 'MSE', 'value': 'mse'},
                        {'label': 'Correlation', 'value': 'correlation'}
                    ],
                    value='final_train_loss'
                )
            ], className='six columns')
        ], className='row'),
        
        html.Div([
            dcc.Graph(id='hyperparams-plot')
        ], style={'marginTop': '20px'})
    ])


def create_statistics_tab():
    """Create statistical summary tab."""
    return html.Div([
        html.H4("Statistical Summary"),
        html.Div([
            html.H5("Performance Statistics"),
            html.Div(id='performance-stats-table')
        ]),
        
        html.Div([
            html.H5("Best Performing Models"),
            html.Div(id='best-models-table')
        ], style={'marginTop': '20px'})
    ])


# Callbacks
@app.callback(
    [Output('load-status', 'children'),
     Output('results-data-store', 'data'),
     Output('experiments-table-container', 'children'),
     Output('training-experiments-dropdown', 'options')],
    [Input('load-results-btn', 'n_clicks')],
    [State('results-dir-input', 'value')]
)
def load_results_callback(n_clicks, results_dir):
    """Load results from directory."""
    if n_clicks is None:
        return "Enter results directory and click Load Results", {}, html.Div(), []
    
    global results_data, experiments_df
    
    try:
        results_data = load_results_from_directory(results_dir)
        
        if not results_data:
            return f"❌ No results found in {results_dir}", {}, html.Div(), []
        
        experiments_df = create_experiments_dataframe(results_data)
        
        # Create experiments table
        table = dash_table.DataTable(
            data=experiments_df.to_dict('records'),
            columns=[{"name": i, "id": i} for i in experiments_df.columns],
            style_cell={'textAlign': 'left'},
            style_data_conditional=[
                {
                    'if': {'row_index': 'odd'},
                    'backgroundColor': 'rgb(248, 248, 248)'
                }
            ],
            style_header={
                'backgroundColor': 'rgb(230, 230, 230)',
                'fontWeight': 'bold'
            },
            page_size=10,
            sort_action="native"
        )
        
        # Create dropdown options
        dropdown_options = [
            {'label': exp, 'value': exp} for exp in experiments_df['experiment'].tolist()
        ]
        
        status = f"✅ Loaded {len(results_data)} experiments from {results_dir}"
        
        return status, results_data, table, dropdown_options
        
    except Exception as e:
        return f"❌ Error loading results: {str(e)}", {}, html.Div(), []


@app.callback(
    Output('analysis-content', 'children'),
    [Input('analysis-tabs', 'value')]
)
def render_analysis_content(active_tab):
    """Render content for active analysis tab."""
    if active_tab == 'comparison':
        return create_comparison_tab()
    elif active_tab == 'training':
        return create_training_tab()
    elif active_tab == 'hyperparams':
        return create_hyperparams_tab()
    elif active_tab == 'statistics':
        return create_statistics_tab()
    else:
        return html.Div("Select a tab to view analysis")


def main():
    """Main function to run the dashboard."""
    parser = argparse.ArgumentParser(description='NeuroBM Results Analyzer Dashboard')
    parser.add_argument('--port', type=int, default=8052, help='Port to run dashboard on')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Host to run dashboard on')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    parser.add_argument('--results_dir', type=str, default='results/', help='Directory containing results')
    
    args = parser.parse_args()
    
    # Pre-load results if directory provided
    if args.results_dir and Path(args.results_dir).exists():
        global results_data, experiments_df
        results_data = load_results_from_directory(args.results_dir)
        if results_data:
            experiments_df = create_experiments_dataframe(results_data)
            logger.info(f"Pre-loaded {len(results_data)} experiments")
    
    # Set up the app layout
    app.layout = create_layout()
    
    # Run the app
    logger.info(f"Starting NeuroBM Results Analyzer on http://{args.host}:{args.port}")
    app.run_server(host=args.host, port=args.port, debug=args.debug)


if __name__ == '__main__':
    main()
