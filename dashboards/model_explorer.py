#!/usr/bin/env python3
"""
Interactive Model Exploration Dashboard for NeuroBM.

This dashboard provides comprehensive visualization and analysis tools
for trained Boltzmann machine models.

Features:
- Model architecture visualization
- Weight matrix heatmaps
- Feature importance analysis
- Latent space exploration
- Sample generation interface
- Interpretability tools

Usage:
    python dashboards/model_explorer.py --checkpoint=path/to/model.pth
    python dashboards/model_explorer.py --port=8051
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import json

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
from neurobm.data.schema import get_schema
from neurobm.interpret.saliency import SaliencyAnalyzer
from neurobm.interpret.mutual_info import MutualInformationAnalyzer
from neurobm.interpret.traversals import LatentTraverser
from neurobm.interpret.tiles import FilterVisualizer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
app = dash.Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])
current_model = None
current_regime = None
feature_names = []


def load_model(checkpoint_path: str) -> torch.nn.Module:
    """Load model from checkpoint."""
    try:
        if checkpoint_path.endswith('.pth'):
            # Try to load different model types
            for model_class in [RestrictedBoltzmannMachine, DeepBoltzmannMachine, ConditionalRBM]:
                try:
                    model = model_class.load_checkpoint(checkpoint_path)
                    logger.info(f"Loaded {model_class.__name__} from {checkpoint_path}")
                    return model
                except:
                    continue
        
        raise ValueError("Could not load model from checkpoint")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return None


def create_layout():
    """Create the dashboard layout."""
    return html.Div([
        # Header
        html.Div([
            html.H1("🔍 NeuroBM Model Explorer", 
                   style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': 20}),
            html.P("Interactive exploration and analysis of trained Boltzmann machines",
                   style={'textAlign': 'center', 'color': '#7f8c8d', 'fontSize': 16})
        ], style={'backgroundColor': '#ecf0f1', 'padding': '20px', 'marginBottom': '20px'}),
        
        # Model Loading Panel
        html.Div([
            html.H3("📁 Model Loading"),
            html.Div([
                html.Div([
                    html.Label("Model Checkpoint Path:"),
                    dcc.Input(
                        id='checkpoint-path-input',
                        type='text',
                        placeholder='path/to/model.pth',
                        style={'width': '100%'}
                    )
                ], className='eight columns'),
                
                html.Div([
                    html.Button('🔄 Load Model', id='load-model-btn', 
                               className='button-primary', style={'marginTop': '25px'})
                ], className='four columns')
            ], className='row'),
            
            html.Div(id='model-load-status', style={'marginTop': '10px'})
        ], style={'backgroundColor': '#f8f9fa', 'padding': '20px', 'marginBottom': '20px'}),
        
        # Model Information Panel
        html.Div([
            html.H3("ℹ️ Model Information"),
            html.Div(id='model-info-panel', children="No model loaded")
        ], style={'backgroundColor': '#e8f5e8', 'padding': '15px', 'marginBottom': '20px'}),
        
        # Visualization Tabs
        html.Div([
            dcc.Tabs(id='visualization-tabs', value='architecture', children=[
                dcc.Tab(label='🏗️ Architecture', value='architecture'),
                dcc.Tab(label='🎯 Weights', value='weights'),
                dcc.Tab(label='🧭 Features', value='features'),
                dcc.Tab(label='🌌 Latent Space', value='latent'),
                dcc.Tab(label='🎲 Sampling', value='sampling')
            ]),
            
            html.Div(id='tab-content')
        ]),
        
        # Store for model data
        dcc.Store(id='model-data-store')
    ])


def create_architecture_tab():
    """Create architecture visualization tab."""
    return html.Div([
        html.H4("Model Architecture"),
        html.Div([
            html.Div([
                dcc.Graph(id='architecture-plot')
            ], className='twelve columns')
        ], className='row'),
        
        html.Div([
            html.H5("Architecture Details"),
            html.Div(id='architecture-details')
        ], style={'marginTop': '20px'})
    ])


def create_weights_tab():
    """Create weights visualization tab."""
    return html.Div([
        html.H4("Weight Analysis"),
        html.Div([
            html.Div([
                dcc.Graph(id='weight-heatmap')
            ], className='six columns'),
            
            html.Div([
                dcc.Graph(id='weight-distribution')
            ], className='six columns')
        ], className='row'),
        
        html.Div([
            html.Div([
                dcc.Graph(id='weight-statistics')
            ], className='twelve columns')
        ], className='row', style={'marginTop': '20px'})
    ])


def create_features_tab():
    """Create feature analysis tab."""
    return html.Div([
        html.H4("Feature Importance Analysis"),
        html.Div([
            html.Div([
                dcc.Graph(id='feature-importance-plot')
            ], className='six columns'),
            
            html.Div([
                dcc.Graph(id='mutual-information-plot')
            ], className='six columns')
        ], className='row'),
        
        html.Div([
            html.H5("Feature Correlations"),
            dcc.Graph(id='feature-correlation-plot')
        ], style={'marginTop': '20px'})
    ])


def create_latent_tab():
    """Create latent space exploration tab."""
    return html.Div([
        html.H4("Latent Space Exploration"),
        html.Div([
            html.Div([
                html.Label("Hidden Unit:"),
                dcc.Dropdown(id='hidden-unit-dropdown', placeholder="Select hidden unit")
            ], className='four columns'),
            
            html.Div([
                html.Label("Traversal Steps:"),
                dcc.Slider(id='traversal-steps-slider', min=5, max=20, value=10, step=1,
                          marks={i: str(i) for i in range(5, 21, 5)})
            ], className='four columns'),
            
            html.Div([
                html.Button('🚀 Generate Traversal', id='generate-traversal-btn', 
                           className='button-primary', style={'marginTop': '25px'})
            ], className='four columns')
        ], className='row'),
        
        html.Div([
            dcc.Graph(id='latent-traversal-plot')
        ], style={'marginTop': '20px'})
    ])


def create_sampling_tab():
    """Create sampling interface tab."""
    return html.Div([
        html.H4("Sample Generation"),
        html.Div([
            html.Div([
                html.Label("Number of Samples:"),
                dcc.Slider(id='n-samples-slider', min=10, max=100, value=50, step=10,
                          marks={i: str(i) for i in range(10, 101, 20)})
            ], className='four columns'),
            
            html.Div([
                html.Label("Temperature:"),
                dcc.Slider(id='temperature-slider', min=0.1, max=2.0, value=1.0, step=0.1,
                          marks={i: f'{i:.1f}' for i in [0.1, 0.5, 1.0, 1.5, 2.0]})
            ], className='four columns'),
            
            html.Div([
                html.Button('🎲 Generate Samples', id='generate-samples-btn', 
                           className='button-primary', style={'marginTop': '25px'})
            ], className='four columns')
        ], className='row'),
        
        html.Div([
            html.Div([
                dcc.Graph(id='generated-samples-plot')
            ], className='six columns'),
            
            html.Div([
                dcc.Graph(id='sample-statistics-plot')
            ], className='six columns')
        ], className='row', style={'marginTop': '20px'})
    ])


# Callbacks
@app.callback(
    [Output('model-load-status', 'children'),
     Output('model-info-panel', 'children'),
     Output('model-data-store', 'data')],
    [Input('load-model-btn', 'n_clicks')],
    [State('checkpoint-path-input', 'value')]
)
def load_model_callback(n_clicks, checkpoint_path):
    """Load model from checkpoint path."""
    if n_clicks is None or not checkpoint_path:
        return "Enter checkpoint path and click Load Model", "No model loaded", {}
    
    global current_model, feature_names
    
    try:
        current_model = load_model(checkpoint_path)
        
        if current_model is None:
            return "❌ Failed to load model", "No model loaded", {}
        
        # Extract model information
        model_info = {
            'type': type(current_model).__name__,
            'n_visible': getattr(current_model, 'n_visible', 'N/A'),
            'n_hidden': getattr(current_model, 'n_hidden', 'N/A'),
            'parameters': sum(p.numel() for p in current_model.parameters()),
            'device': str(current_model.device)
        }
        
        # Try to get feature names from regime
        try:
            if hasattr(current_model, 'regime'):
                schema = get_schema(current_model.regime)
                feature_names = list(schema.features.keys())
            else:
                feature_names = [f'Feature_{i}' for i in range(model_info['n_visible'])]
        except:
            feature_names = [f'Feature_{i}' for i in range(model_info['n_visible'])]
        
        status = f"✅ Successfully loaded {model_info['type']}"
        
        info_text = f"""
        **Model Type:** {model_info['type']}
        **Visible Units:** {model_info['n_visible']}
        **Hidden Units:** {model_info['n_hidden']}
        **Total Parameters:** {model_info['parameters']:,}
        **Device:** {model_info['device']}
        **Features:** {', '.join(feature_names[:5])}{'...' if len(feature_names) > 5 else ''}
        """
        
        return status, info_text, model_info
        
    except Exception as e:
        return f"❌ Error loading model: {str(e)}", "No model loaded", {}


@app.callback(
    Output('tab-content', 'children'),
    [Input('visualization-tabs', 'value')]
)
def render_tab_content(active_tab):
    """Render content for active tab."""
    if active_tab == 'architecture':
        return create_architecture_tab()
    elif active_tab == 'weights':
        return create_weights_tab()
    elif active_tab == 'features':
        return create_features_tab()
    elif active_tab == 'latent':
        return create_latent_tab()
    elif active_tab == 'sampling':
        return create_sampling_tab()
    else:
        return html.Div("Select a tab to view content")


def main():
    """Main function to run the dashboard."""
    parser = argparse.ArgumentParser(description='NeuroBM Model Explorer Dashboard')
    parser.add_argument('--port', type=int, default=8051, help='Port to run dashboard on')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Host to run dashboard on')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    parser.add_argument('--checkpoint', type=str, help='Path to model checkpoint to load')
    
    args = parser.parse_args()
    
    # Load model if checkpoint provided
    if args.checkpoint:
        global current_model
        current_model = load_model(args.checkpoint)
        if current_model:
            logger.info(f"Pre-loaded model from {args.checkpoint}")
    
    # Set up the app layout
    app.layout = create_layout()
    
    # Run the app
    logger.info(f"Starting NeuroBM Model Explorer on http://{args.host}:{args.port}")
    app.run_server(host=args.host, port=args.port, debug=args.debug)


if __name__ == '__main__':
    main()
