#!/usr/bin/env python3
"""
Interactive Training Monitoring Dashboard for NeuroBM.

This dashboard provides real-time monitoring of Boltzmann machine training
with interactive visualizations and controls.

Features:
- Real-time training metrics visualization
- Model architecture visualization
- Weight distribution analysis
- Interactive parameter controls
- Export capabilities

Usage:
    python dashboards/training_monitor.py --port=8050
    python dashboards/training_monitor.py --config=experiments/base.yaml
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import yaml
import json
from datetime import datetime

import dash
from dash import dcc, html, Input, Output, State, callback
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
import torch

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from neurobm.models.rbm import RestrictedBoltzmannMachine
from neurobm.models.dbm import DeepBoltzmannMachine
from neurobm.models.crbm import ConditionalRBM
from neurobm.data.loaders import get_data_loader
from neurobm.training.loop import TrainingLoop
from neurobm.training.callbacks import get_standard_callbacks

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for dashboard state
app = dash.Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])
training_data = {
    'epochs': [],
    'train_loss': [],
    'val_loss': [],
    'reconstruction_error': [],
    'free_energy': [],
    'weight_norm': []
}
current_model = None
current_config = None


def create_layout():
    """Create the dashboard layout."""
    return html.Div([
        # Header
        html.Div([
            html.H1("🧠 NeuroBM Training Monitor", 
                   style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': 30}),
            html.P("Real-time monitoring of Boltzmann machine training",
                   style={'textAlign': 'center', 'color': '#7f8c8d', 'fontSize': 18})
        ], style={'backgroundColor': '#ecf0f1', 'padding': '20px', 'marginBottom': '20px'}),
        
        # Control Panel
        html.Div([
            html.H3("🎛️ Control Panel"),
            html.Div([
                html.Div([
                    html.Label("Model Type:"),
                    dcc.Dropdown(
                        id='model-type-dropdown',
                        options=[
                            {'label': 'RBM', 'value': 'rbm'},
                            {'label': 'DBM', 'value': 'dbm'},
                            {'label': 'CRBM', 'value': 'crbm'}
                        ],
                        value='rbm'
                    )
                ], className='three columns'),
                
                html.Div([
                    html.Label("Hidden Units:"),
                    dcc.Slider(
                        id='hidden-units-slider',
                        min=16, max=512, step=16, value=128,
                        marks={i: str(i) for i in range(16, 513, 64)},
                        tooltip={"placement": "bottom", "always_visible": True}
                    )
                ], className='three columns'),
                
                html.Div([
                    html.Label("Learning Rate:"),
                    dcc.Slider(
                        id='lr-slider',
                        min=0.001, max=0.1, step=0.001, value=0.01,
                        marks={0.001: '0.001', 0.01: '0.01', 0.05: '0.05', 0.1: '0.1'},
                        tooltip={"placement": "bottom", "always_visible": True}
                    )
                ], className='three columns'),
                
                html.Div([
                    html.Button('🚀 Start Training', id='start-training-btn', 
                               className='button-primary', style={'marginTop': '25px'}),
                    html.Button('⏹️ Stop Training', id='stop-training-btn', 
                               className='button', style={'marginTop': '25px', 'marginLeft': '10px'})
                ], className='three columns')
            ], className='row')
        ], style={'backgroundColor': '#f8f9fa', 'padding': '20px', 'marginBottom': '20px'}),
        
        # Training Status
        html.Div([
            html.H3("📊 Training Status"),
            html.Div(id='training-status', children="Ready to start training...")
        ], style={'backgroundColor': '#e8f5e8', 'padding': '15px', 'marginBottom': '20px'}),
        
        # Metrics Dashboard
        html.Div([
            html.H3("📈 Training Metrics"),
            html.Div([
                html.Div([
                    dcc.Graph(id='loss-plot')
                ], className='six columns'),
                
                html.Div([
                    dcc.Graph(id='metrics-plot')
                ], className='six columns')
            ], className='row'),
            
            html.Div([
                html.Div([
                    dcc.Graph(id='weight-distribution-plot')
                ], className='six columns'),
                
                html.Div([
                    dcc.Graph(id='model-architecture-plot')
                ], className='six columns')
            ], className='row')
        ]),
        
        # Model Information
        html.Div([
            html.H3("🔍 Model Information"),
            html.Div(id='model-info', children="No model loaded")
        ], style={'backgroundColor': '#f0f8ff', 'padding': '15px', 'marginTop': '20px'}),
        
        # Auto-refresh interval
        dcc.Interval(
            id='interval-component',
            interval=2000,  # Update every 2 seconds
            n_intervals=0
        ),
        
        # Store for training data
        dcc.Store(id='training-data-store', data=training_data)
    ])


def create_loss_plot(data: Dict[str, List]) -> go.Figure:
    """Create training loss plot."""
    fig = go.Figure()
    
    if data['epochs']:
        fig.add_trace(go.Scatter(
            x=data['epochs'],
            y=data['train_loss'],
            mode='lines+markers',
            name='Training Loss',
            line=dict(color='#e74c3c', width=2)
        ))
        
        if data['val_loss']:
            fig.add_trace(go.Scatter(
                x=data['epochs'],
                y=data['val_loss'],
                mode='lines+markers',
                name='Validation Loss',
                line=dict(color='#3498db', width=2)
            ))
    
    fig.update_layout(
        title='Training Loss Over Time',
        xaxis_title='Epoch',
        yaxis_title='Loss',
        hovermode='x unified',
        template='plotly_white'
    )
    
    return fig


def create_metrics_plot(data: Dict[str, List]) -> go.Figure:
    """Create additional metrics plot."""
    fig = go.Figure()
    
    if data['epochs']:
        fig.add_trace(go.Scatter(
            x=data['epochs'],
            y=data['reconstruction_error'],
            mode='lines+markers',
            name='Reconstruction Error',
            line=dict(color='#f39c12', width=2)
        ))
        
        if data['free_energy']:
            fig.add_trace(go.Scatter(
                x=data['epochs'],
                y=data['free_energy'],
                mode='lines+markers',
                name='Free Energy',
                line=dict(color='#9b59b6', width=2),
                yaxis='y2'
            ))
    
    fig.update_layout(
        title='Training Metrics',
        xaxis_title='Epoch',
        yaxis_title='Reconstruction Error',
        yaxis2=dict(
            title='Free Energy',
            overlaying='y',
            side='right'
        ),
        hovermode='x unified',
        template='plotly_white'
    )
    
    return fig


def create_weight_distribution_plot(model) -> go.Figure:
    """Create weight distribution plot."""
    fig = go.Figure()
    
    if model is not None:
        try:
            weights = model.W.detach().cpu().numpy().flatten()
            
            fig.add_trace(go.Histogram(
                x=weights,
                nbinsx=50,
                name='Weight Distribution',
                marker_color='#2ecc71'
            ))
            
            # Add statistics
            mean_weight = np.mean(weights)
            std_weight = np.std(weights)
            
            fig.add_vline(x=mean_weight, line_dash="dash", line_color="red",
                         annotation_text=f"Mean: {mean_weight:.4f}")
            
        except Exception as e:
            logger.warning(f"Could not plot weight distribution: {e}")
    
    fig.update_layout(
        title='Weight Distribution',
        xaxis_title='Weight Value',
        yaxis_title='Frequency',
        template='plotly_white'
    )
    
    return fig


# Callbacks
@app.callback(
    [Output('loss-plot', 'figure'),
     Output('metrics-plot', 'figure'),
     Output('weight-distribution-plot', 'figure'),
     Output('training-status', 'children'),
     Output('model-info', 'children')],
    [Input('interval-component', 'n_intervals')],
    [State('training-data-store', 'data')]
)
def update_dashboard(n_intervals, stored_data):
    """Update dashboard with latest training data."""
    global current_model, training_data
    
    # Update plots
    loss_fig = create_loss_plot(training_data)
    metrics_fig = create_metrics_plot(training_data)
    weight_fig = create_weight_distribution_plot(current_model)
    
    # Update status
    if training_data['epochs']:
        last_epoch = training_data['epochs'][-1]
        last_loss = training_data['train_loss'][-1]
        status = f"Epoch {last_epoch}: Loss = {last_loss:.4f}"
    else:
        status = "Ready to start training..."
    
    # Update model info
    if current_model is not None:
        model_info = f"""
        Model Type: {type(current_model).__name__}
        Visible Units: {current_model.n_visible}
        Hidden Units: {current_model.n_hidden}
        Parameters: {sum(p.numel() for p in current_model.parameters())}
        Device: {current_model.device}
        """
    else:
        model_info = "No model loaded"
    
    return loss_fig, metrics_fig, weight_fig, status, model_info


def main():
    """Main function to run the dashboard."""
    parser = argparse.ArgumentParser(description='NeuroBM Training Monitor Dashboard')
    parser.add_argument('--port', type=int, default=8050, help='Port to run dashboard on')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Host to run dashboard on')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    parser.add_argument('--config', type=str, help='Path to experiment config file')
    
    args = parser.parse_args()
    
    # Load config if provided
    if args.config:
        try:
            with open(args.config, 'r') as f:
                global current_config
                current_config = yaml.safe_load(f)
            logger.info(f"Loaded config from {args.config}")
        except Exception as e:
            logger.error(f"Could not load config: {e}")
    
    # Set up the app layout
    app.layout = create_layout()
    
    # Run the app
    logger.info(f"Starting NeuroBM Training Monitor on http://{args.host}:{args.port}")
    app.run_server(host=args.host, port=args.port, debug=args.debug)


if __name__ == '__main__':
    main()
