import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import base64
import cv2
import numpy as np
from skimage import exposure
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from skimage.util import random_noise
from scipy.signal import convolve2d
from dash import callback_context

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Function to decode uploaded image
def parse_image(contents):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    np_arr = np.frombuffer(decoded, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    return None

# Log transformation
def log_transformation(image, factor=1):
    # Check if the image is grayscale or color
    if len(image.shape) == 2:
        # Grayscale image
        c = 255 / (np.log(1 + np.max(image)))
        log_image = c * (np.log(1 + image) ** factor)
        return np.array(log_image, dtype=np.uint8)
    else:
        # Color image
        log_image = np.zeros_like(image, dtype=np.float64)
        for i in range(3):  # process each channel
            channel = image[:,:,i]
            c = 255 / (np.log(1 + np.max(channel)))
            log_image[:,:,i] = c * (np.log(1 + channel) ** factor)
        return np.array(log_image, dtype=np.uint8)

# Power Law (Gamma) transformation
def gamma_transformation(image, gamma=1.0):
    gamma_corrected = np.array(255 * (image / 255) ** gamma, dtype=np.uint8)
    return gamma_corrected

# Contrast stretching
def contrast_stretching(image, percentile_range):
    p_low, p_high = percentile_range
    p_low, p_high = np.clip(p_low, 0, 100), np.clip(p_high, 0, 100)
    if len(image.shape) == 2:  # Grayscale
        p2, p98 = np.percentile(image, (p_low, p_high))
        return exposure.rescale_intensity(image, in_range=(p2, p98))
    else:  # Color
        img_rescaled = np.zeros_like(image)
        for i in range(3):
            p2, p98 = np.percentile(image[:,:,i], (p_low, p_high))
            img_rescaled[:,:,i] = exposure.rescale_intensity(image[:,:,i], in_range=(p2, p98))
        return img_rescaled

# Histogram equalization
def histogram_equalization(image, clip_limit):
    if len(image.shape) == 2:  # Grayscale
        return exposure.equalize_adapthist(image, clip_limit=clip_limit)
    else:  # Color
        img_eq = np.zeros_like(image, dtype=np.float64)
        for i in range(3):
            img_eq[:,:,i] = exposure.equalize_adapthist(image[:,:,i], clip_limit=clip_limit)
        return (img_eq * 255).astype(np.uint8)

# Image Quantization
def image_quantization(image, n_colors=8):
    # Reshape the image to a 2D array of pixels
    pixels = image.reshape(-1, 3)
    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(pixels)
    # Replace each pixel with its nearest cluster center
    new_pixels = kmeans.cluster_centers_[kmeans.labels_]
    # Reshape back to the original image shape
    quantized_image = new_pixels.reshape(image.shape).astype(np.uint8)
    return quantized_image

# Function to calculate histogram
def compute_histogram(image):
    if len(image.shape) == 2:  # Grayscale image
        hist = np.histogram(image, bins=256, range=(0, 256))[0]
        return hist
    else:
        hist_red = np.histogram(image[..., 0], bins=256, range=(0, 256))[0]
        hist_green = np.histogram(image[..., 1], bins=256, range=(0, 256))[0]
        hist_blue = np.histogram(image[..., 2], bins=256, range=(0, 256))[0]
        return hist_red, hist_green, hist_blue

# New function for image padding
def image_padding(image, pad_width, pad_color=[255, 0, 0]):  # Default to red padding
    # Ensure pad_color is in BGR format for OpenCV
    pad_color_bgr = pad_color[::-1]
    
    return cv2.copyMakeBorder(
        image,
        top=pad_width,
        bottom=pad_width,
        left=pad_width,
        right=pad_width,
        borderType=cv2.BORDER_CONSTANT,
        value=pad_color_bgr
    )

# Add this new function for adding noise to an image
def add_noise(image, noise_type='gaussian', variance=0.01):
    if noise_type == 'gaussian':
        noisy_image = random_noise(image, mode=noise_type, var=variance)
    else:
        # For other noise types, you might need to adjust parameters accordingly
        noisy_image = random_noise(image, mode=noise_type)
    
    return np.clip(noisy_image * 255, 0, 255).astype(np.uint8)

# Add this new function for convolution
def apply_convolution(image, kernel):
    if len(image.shape) == 2:  # Grayscale image
        return cv2.filter2D(image, -1, kernel)
    else:  # Color image
        result = np.zeros_like(image)
        for i in range(3):
            result[:,:,i] = cv2.filter2D(image[:,:,i], -1, kernel)
        return result

# Define some common kernels
kernels = {
    'blur': np.ones((5,5)) / 25,
    'sharpen': np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]]),
    'edge_detect': np.array([[-1,-1,-1], [-1,8,-1], [-1,-1,-1]]),
    'emboss': np.array([[-2,-1,0], [-1,1,1], [0,1,2]])
}

# Layout of the app
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Image Processing Dashboard"), className="mb-2")
    ]),
    dbc.Row([
        dbc.Col([
            dcc.Upload(id='upload-image', children=html.Div(['Drag and Drop or ', html.A('Select Image')]),
                       style={'width': '100%', 'height': '60px', 'lineHeight': '60px',
                              'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px',
                              'textAlign': 'center', 'margin': '10px'}),
            dcc.Upload(id='upload-reference',
                       children=html.Div(['Drag and Drop or ', html.A('Select Reference Image')]),
                       style={'width': '100%', 'height': '60px', 'lineHeight': '60px',
                              'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px',
                              'textAlign': 'center', 'margin': '10px'}, disabled=True),
            dcc.Dropdown(id='transformation-type',
                         options=[
                             {'label': 'Log Transformation', 'value': 'log'},
                             {'label': 'Power Law Transformation (Gamma)', 'value': 'gamma'},
                             {'label': 'Contrast Stretching', 'value': 'contrast'},
                             {'label': 'Histogram Equalization', 'value': 'hist_eq'},
                             {'label': 'Histogram Matching', 'value': 'hist_match'},
                             {'label': 'Image Quantization', 'value': 'quantization'},
                             {'label': 'Image Padding', 'value': 'padding'},
                             {'label': 'Add Noise', 'value': 'noise'},
                             {'label': 'Convolution', 'value': 'convolution'}  # New option
                         ],
                         value='log',
                         clearable=False,
                         style={'margin-bottom': '20px'}
                         ),
            dcc.Dropdown(id='convolution-kernel',
                         options=[
                             {'label': 'Blur', 'value': 'blur'},
                             {'label': 'Sharpen', 'value': 'sharpen'},
                             {'label': 'Edge Detect', 'value': 'edge_detect'},
                             {'label': 'Emboss', 'value': 'emboss'}
                         ],
                         value='blur',
                         clearable=False,
                         style={'margin-bottom': '20px', 'display': 'none'}
                         ),
            html.Div([
                dcc.Slider(
                    id='gamma-slider',
                    min=0.1,
                    max=5,
                    step=0.1,
                    value=1,
                    marks={i: str(i) for i in range(6)},
                    tooltip={"placement": "bottom", "always_visible": True},
                    updatemode='drag'
                )
            ], style={'margin-bottom': '10px'}),
            html.Div("Adjustment slider (used for multiple transformations)", style={'margin-bottom': '20px', 'font-style': 'italic'}),
            html.Div([
                dcc.Slider(
                    id='quantization-slider',
                    min=2,
                    max=64,
                    step=1,
                    value=8, 
                    marks={2: '2', 32: '32', 64: '64'},
                    tooltip={"placement": "bottom", "always_visible": True},
                    updatemode='drag'
                )
            ], style={'margin-bottom': '10px'}),
            html.Div("Quantization slider (for Image Quantization only)", style={'margin-bottom': '20px', 'font-style': 'italic'}),
            html.Div([
                dcc.Slider(
                    id='padding-slider',
                    min=0,
                    max=100,
                    step=1,
                    value=10,
                    marks={0: '0', 50: '50', 100: '100'},
                    tooltip={"placement": "bottom", "always_visible": True},
                    updatemode='drag'
                )
            ], style={'margin-bottom': '10px'}),
            html.Div("Padding slider (for Image Padding only)", style={'margin-bottom': '20px', 'font-style': 'italic'}),
        ], width=4),
        dbc.Col([
            html.Div(id='output-original-image', style={'margin-bottom': '20px'}),
            html.Div(id='output-transformed-image'),
            html.Div(id='output-original-histogram', style={'margin-bottom': '20px'}),
            html.Div(id='output-transformed-histogram')
        ], width=8)
    ])
], fluid=True)

# Add this function
def histogram_matching(image, reference):
    # Check if images are grayscale or color
    if len(image.shape) == 2:
        # Grayscale images
        matched = exposure.match_histograms(image, reference)
    else:
        # Color images
        matched = np.zeros_like(image)
        for i in range(3):  # process each channel
            matched[:,:,i] = exposure.match_histograms(image[:,:,i], reference[:,:,i])
    
    return matched.astype(np.uint8)

# Callbacks for transformations and enabling/disabling reference upload
@app.callback(
    Output('output-original-image', 'children'),
    Output('output-transformed-image', 'children'),
    Output('output-original-histogram', 'children'),
    Output('output-transformed-histogram', 'children'),
    Output('gamma-slider', 'disabled'),
    Output('quantization-slider', 'disabled'),
    Output('padding-slider', 'disabled'),
    Output('upload-reference', 'disabled'),
    Output('convolution-kernel', 'style'),
    Input('upload-image', 'contents'),
    Input('upload-reference', 'contents'),
    Input('transformation-type', 'value'),
    Input('gamma-slider', 'value'),
    Input('quantization-slider', 'value'),
    Input('padding-slider', 'value'),
    Input('convolution-kernel', 'value')
)
def update_output(contents, reference_contents, transformation_type, gamma_value, quantization_value, padding_value, convolution_kernel):
    ctx = callback_context
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    convolution_dropdown_style = {'margin-bottom': '20px', 'display': 'none'}

    if trigger_id == 'transformation-type':
        if transformation_type == 'convolution':
            convolution_dropdown_style = {'margin-bottom': '20px', 'display': 'block'}
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, convolution_dropdown_style

    if contents is None:
        return "Upload an image to start!", None, None, None, True, True, True, True, convolution_dropdown_style

    # Parse the uploaded image
    image = parse_image(contents)
    if image is None:
        return "Invalid image file!", None, None, None, True, True, True, True, convolution_dropdown_style

    # Initialize variables
    transformed_image = None
    disable_gamma_slider = True
    disable_quantization_slider = True
    disable_padding_slider = True
    reference_required = transformation_type == 'hist_match'

    if reference_required:
        if reference_contents is None:
            return "Upload a reference image to proceed with histogram matching!", None, None, None, True, True, True, False, {'margin-bottom': '20px', 'display': 'none'}
        reference_image = parse_image(reference_contents)
        if reference_image is None:
            return "Invalid reference image file!", None, None, None, True, True, True, False, {'margin-bottom': '20px', 'display': 'none'}
        transformed_image = histogram_matching(image, reference_image)
    else:
        if transformation_type == 'log':
            transformed_image = log_transformation(image, gamma_value)
            disable_gamma_slider = False
        elif transformation_type == 'gamma':
            transformed_image = gamma_transformation(image, gamma_value)
            disable_gamma_slider = False
        elif transformation_type == 'contrast':
            percentile_range = (gamma_value, 100 - gamma_value)
            transformed_image = contrast_stretching(image, percentile_range)
            disable_gamma_slider = False
        elif transformation_type == 'hist_eq':
            clip_limit = gamma_value / 10
            transformed_image = histogram_equalization(image, clip_limit)
            disable_gamma_slider = False
        elif transformation_type == 'quantization':
            transformed_image = image_quantization(image, quantization_value)
            disable_quantization_slider = False
        elif transformation_type == 'padding':
            transformed_image = image_padding(image, padding_value, pad_color=[0, 0, 255])
            disable_padding_slider = False
        elif transformation_type == 'noise':
            variance = (gamma_value - 0.1) / 49
            transformed_image = add_noise(image, variance=variance)
            disable_gamma_slider = False
        elif transformation_type == 'convolution':
            kernel = kernels[convolution_kernel]
            transformed_image = apply_convolution(image, kernel)
            convolution_dropdown_style = {'margin-bottom': '20px', 'display': 'block'}

    # Plotly figure for original image
    fig_original = go.Figure(go.Image(z=image))
    fig_original.update_layout(title="Original Image", coloraxis_showscale=False, margin=dict(l=0, r=0, t=30, b=0))

    # Plotly figure for transformed image
    fig_transformed = go.Figure(go.Image(z=transformed_image))
    fig_transformed.update_layout(title="Transformed Image", coloraxis_showscale=False,
                                  margin=dict(l=0, r=0, t=30, b=0))

    # Compute histograms
    original_hist = compute_histogram(image)
    transformed_hist = compute_histogram(transformed_image)

    # Plot histogram for original image
    if len(image.shape) == 2:  # Grayscale
        fig_original_hist = go.Figure(go.Bar(y=original_hist))
    else:  # RGB
        fig_original_hist = go.Figure()
        fig_original_hist.add_trace(go.Bar(y=original_hist[0], name='Red'))
        fig_original_hist.add_trace(go.Bar(y=original_hist[1], name='Green'))
        fig_original_hist.add_trace(go.Bar(y=original_hist[2], name='Blue'))
    fig_original_hist.update_layout(title="Original Histogram", barmode='overlay', margin=dict(l=0, r=0, t=30, b=0))

    # Plot histogram for transformed image
    if len(transformed_image.shape) == 2:  # Grayscale
        fig_transformed_hist = go.Figure(go.Bar(y=transformed_hist))
    else:  # RGB
        fig_transformed_hist = go.Figure()
        fig_transformed_hist.add_trace(go.Bar(y=transformed_hist[0], name='Red'))
        fig_transformed_hist.add_trace(go.Bar(y=transformed_hist[1], name='Green'))
        fig_transformed_hist.add_trace(go.Bar(y=transformed_hist[2], name='Blue'))
    fig_transformed_hist.update_layout(title="Transformed Histogram", barmode='overlay',
                                       margin=dict(l=0, r=0, t=30, b=0))

    # Update the return statement
    return (
        dcc.Graph(figure=fig_original),
        dcc.Graph(figure=fig_transformed),
        dcc.Graph(figure=fig_original_hist),
        dcc.Graph(figure=fig_transformed_hist),
        disable_gamma_slider,
        disable_quantization_slider,
        disable_padding_slider,
        reference_required,
        convolution_dropdown_style
    )

if __name__ == '__main__':
    app.run_server(debug=True)
# Comment
