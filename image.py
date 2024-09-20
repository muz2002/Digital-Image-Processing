import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import base64
import cv2
import numpy as np
from skimage import exposure
import plotly.graph_objects as go


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
def log_transformation(image):
    c = 255 / np.log(1 + np.max(image))
    log_image = c * np.log(1 + image)
    return np.array(log_image, dtype=np.uint8)


# Power Law (Gamma) transformation
def gamma_transformation(image, gamma=1.0):
    gamma_corrected = np.array(255 * (image / 255) ** gamma, dtype=np.uint8)
    return gamma_corrected


# Contrast stretching
def contrast_stretching(image):
    p2, p98 = np.percentile(image, (2, 98))
    img_rescale = exposure.rescale_intensity(image, in_range=(p2, p98))
    return img_rescale


# Histogram equalization
def histogram_equalization(image):
    gray_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    img_eq = cv2.equalizeHist(gray_img)
    img_eq_rgb = cv2.cvtColor(img_eq, cv2.COLOR_GRAY2RGB)
    return img_eq_rgb


# Histogram matching (using a reference image)
def histogram_matching(image, reference):
    # Split the image into its respective channels
    matched = np.zeros_like(image)
    for c in range(image.shape[2]):
        matched[..., c] = exposure.match_histograms(image[..., c], reference[..., c])
    return matched


# Function to calculate histogram
def compute_histogram(image):
    # Compute histogram for each channel separately
    if len(image.shape) == 2:  # Grayscale image
        hist = np.histogram(image, bins=256, range=(0, 256))[0]
        return hist
    else:
        hist_red = np.histogram(image[..., 0], bins=256, range=(0, 256))[0]
        hist_green = np.histogram(image[..., 1], bins=256, range=(0, 256))[0]
        hist_blue = np.histogram(image[..., 2], bins=256, range=(0, 256))[0]
        return hist_red, hist_green, hist_blue


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
                             {'label': 'Histogram Matching', 'value': 'hist_match'}
                         ],
                         value='log',
                         clearable=False
                         ),
            dcc.Slider(id='gamma-slider', min=0.1, max=5, step=0.1, value=1, marks={i: str(i) for i in range(6)},
                       tooltip={"placement": "bottom", "always_visible": True}, updatemode='drag', included=False),
        ], width=4),
        dbc.Col([
            html.Div(id='output-original-image', style={'margin-bottom': '20px'}),
            html.Div(id='output-transformed-image'),
            html.Div(id='output-original-histogram', style={'margin-bottom': '20px'}),
            html.Div(id='output-transformed-histogram')
        ], width=8)
    ])
], fluid=True)


# Callbacks for transformations and enabling/disabling reference upload
@app.callback(
    Output('output-original-image', 'children'),
    Output('output-transformed-image', 'children'),
    Output('output-original-histogram', 'children'),
    Output('output-transformed-histogram', 'children'),
    Output('gamma-slider', 'included'),
    Output('upload-reference', 'disabled'),
    Input('upload-image', 'contents'),
    Input('upload-reference', 'contents'),
    Input('transformation-type', 'value'),
    Input('gamma-slider', 'value')
)
def update_output(contents, reference_contents, transformation_type, gamma_value):
    if contents is None:
        return "Upload an image to start!", None, None, None, False, True

    # Parse the uploaded image
    image = parse_image(contents)
    if image is None:
        return "Invalid image file!", None, None, None, False, True

    # Initialize variables
    transformed_image = None
    show_gamma_slider = False
    reference_required = transformation_type == 'hist_match'

    # Check if transformation requires a reference image
    if reference_required:
        if reference_contents is None:
            return "Upload a reference image to proceed with histogram matching!", None, None, None, False, False
        reference_image = parse_image(reference_contents)
        if reference_image is None:
            return "Invalid reference image file!", None, None, None, False, False
        transformed_image = histogram_matching(image, reference_image)
    else:
        if transformation_type == 'log':
            transformed_image = log_transformation(image)
        elif transformation_type == 'gamma':
            transformed_image = gamma_transformation(image, gamma_value)
            show_gamma_slider = True
        elif transformation_type == 'contrast':
            transformed_image = contrast_stretching(image)
        elif transformation_type == 'hist_eq':
            transformed_image = histogram_equalization(image)

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

    # Enable or disable the reference image upload based on transformation type
    return (dcc.Graph(figure=fig_original), dcc.Graph(figure=fig_transformed),
            dcc.Graph(figure=fig_original_hist), dcc.Graph(figure=fig_transformed_hist), show_gamma_slider,
            not reference_required)


if __name__ == '__main__':
    app.run_server(debug=True)
