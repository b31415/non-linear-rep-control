from abc import ABC, abstractmethod
import time
import itertools
import math
import psutil
import multiprocessing as mp
import warnings
from typing import Callable
from functools import partial
import numpy as np
from numpy.polynomial.polynomial import Polynomial
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.colors import LinearSegmentedColormap
import scipy.linalg
from sklearn.decomposition import PCA, SparsePCA
from scipy.optimize import minimize_scalar, differential_evolution, lsq_linear, minimize
from scipy.interpolate import RBFInterpolator
import torch
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression, Ridge
from contextlib import nullcontext
import scipy
from .pipelines import RepReadingPipeline
import pickle
from timeit import timeit
from sklearn.model_selection import train_test_split

def get_hidden_states_for_rep_reader(prompts):
    max_honesty = max([float(k) for k in prompts.keys()])
    prompts = [(float(honesty) / max_honesty, prompt) for honesty, prompt in prompts.items()]
    return prompts

def project_onto_curve(H, curve):
    """Project matrix H (n, d_1) onto curve vector (d_2,)"""
    # Calculate the magnitude of the curve vector
     # Ensure H and curve are on the same device (CPU or GPU)
    if type(curve) != torch.Tensor:
        H = torch.Tensor(H).cuda()
    if type(curve) != torch.Tensor:
        curve = torch.Tensor(curve)
        curve = curve.to(H.device)
    mag = torch.norm(curve)
    assert not torch.isinf(mag).any()
    # Calculate the projection
    projection = H.matmul(curve) / mag
    return projection

ABLATE_NON_LINEAR_CONTROL = False
DEFAULT_RBF_KWARGS = dict(
    kernel='inverse_quadratic',#'gaussian',
    epsilon=2.71,#2.5,
    smoothing=0.04   
)
DEFAULT_CONCEPT_PROMPT_TEMPLATE = {
    1:    "Pretend you are the world's most {concept_words[0]} person. Ensure that you are concise and always {concept_words[2]}.",
    0.8:  "Pretend you are an extremely {concept_words[0]} person. Ensure that you are concise and almost always {concept_words[2]}.",
    0.6:  "Pretend you are a highly {concept_words[0]} person. Ensure that you are concise and usually {concept_words[2]}.",
    0.4:  "Pretend you are a person that is quite {concept_words[0]}. Ensure that you are concise and generally, but not always, {concept_words[2]}.",
    0.2:  "Pretend you are a person with a slight preference towards being {concept_words[0]}. Ensure that you are concise and often, but not always, {concept_words[2]}.",
    0:    "Pretend you are neither particularly {concept_words[0]} nor {concept_words[1]}. Ensure that you are concise and regularly {concept_words[2]}, but also regularly don't.",
    -0.2: "Pretend you are a person with a slight preference towards being {concept_words[1]}. Ensure that you are concise and only sometimes {concept_words[2]}.",
    -0.4: "Pretend you are a person that is quite {concept_words[1]}. Ensure that you are concise and {concept_words[2]} only on occasion.",
    -0.6: "Pretend you are a highly {concept_words[1]} person. Ensure that you are concise and rarely {concept_words[2]}.",
    -0.8: "Pretend you are an extremely {concept_words[1]} person. Ensure that you are concise and almost never {concept_words[2]}.",
    -1:   "Pretend you are the world's most {concept_words[1]} person. Ensure that you are concise and never {concept_words[2]} under any circumstance."
}

SHORT_CONCEPT_PROMPT_TEMPLATE = {
    1:    "Pretend you are the world's most {concept_words[0]} person. You always {concept_words[2]}.",
    0.8:  "Pretend you are an extremely {concept_words[0]} person. You almost always {concept_words[2]}.",
    0.6:  "Pretend you are a highly {concept_words[0]} person. You usually {concept_words[2]}.",
    0.4:  "Pretend you are a person that is quite {concept_words[0]}. You generally, but not always, {concept_words[2]}.",
    0.2:  "Pretend you are a person with a slight preference towards being {concept_words[0]}. You more often then not {concept_words[2]}.",
    0:    "Pretend you are neither particularly {concept_words[0]} nor {concept_words[1]}. You regularly {concept_words[2]}, but also regularly don't.",
    -0.2: "Pretend you are a person with a slight preference towards being {concept_words[1]}. You only sometimes {concept_words[2]}.",
    -0.4: "Pretend you are a person that is quite {concept_words[1]}. You {concept_words[2]} only on occasion.",
    -0.6: "Pretend you are a highly {concept_words[1]} person. You rarely {concept_words[2]}.",
    -0.8: "Pretend you are an extremely {concept_words[1]} person. You almost never {concept_words[2]}.",
    -1:   "Pretend you are the world's most {concept_words[1]} person. You never {concept_words[2]} under any circumstance."
}

class NonLinearRepReadingPipeline(RepReadingPipeline):
    
    def __init__(self,
        **kwargs
    ):
        super().__init__(**kwargs)

    def _get_hidden_states(
        self, 
        outputs,
        rep_token: str | int = -1,
        hidden_layers: list[int] | int = -1,
        which_hidden_states: str | None = None
    ):
        """
        Args:
            rep_token: Index of a token relative to the end of the string. The hidden
                state corresponding to this token will be returned. If None, average all tokens
        """
        if hasattr(outputs, 'encoder_hidden_states') and hasattr(outputs, 'decoder_hidden_states'):
            outputs['hidden_states'] = outputs[f'{which_hidden_states}_hidden_states']
    
        hidden_states_layers = {}
        for layer in hidden_layers:
            hidden_states = outputs['hidden_states'][layer]
            hidden_states =  hidden_states[:, rep_token, :]
            hidden_states_layers[layer] = hidden_states.detach()

        return hidden_states_layers
    
    def get_directions(
        self, 
        train_inputs: list[str], 
        direction_method: str,
        train_labels: list[int] = None,
        rep_token: list[int] | int | None = -1, 
        hidden_layers: str | int = -1,
        n_difference: int = 1,
        batch_size: int = 16, 
        direction_finder_kwargs: dict = {},
        which_hidden_states: str | None = None,
        average_rep_tokens: bool = True,
        **tokenizer_args
    ):
        """Train a Rep on the training data.
        Args:
            batch_size: batch size to use when getting hidden states
            direction_method: string specifying the Rep strategy for finding directions
            direction_finder_kwargs: kwargs to pass to Rep constructor
            rep_token: 
                If list, each element of rep_token is the (zero-)index of the token states
                    to start averaging from. E.g. if rep_token[0] = 5, average over the states
                    that produce tokens 5, 6, 7, ... (i.e., collect the states where token 4
                    is the last token of the input prompt, etc.).
                If None, average over all tokens (similarly to list).
                If int, only get the direction using the nth token from the end for all inputs.
            average_rep_tokens:
                If True, then the activations for multiple tokens will be average as described for
                    `rep_token', resulting in the `hidden_states' passed to get_directions having shape
                    {n_layers (int): [len(train_inputs), n_dims] (Tensor)}.
                If False, the states for each token will not be averaged, giving the form 
                {n_layers: [len(train_inputs), n_dims, n_tokens]}.
                Default: True.
        """
        if not average_rep_tokens and direction_method != 'least-squares':
            raise NotImplementedError(f"Only 'least-squares supports average_rep_tokens=False, but got method {direction_method}.")
        
        if not isinstance(hidden_layers, list): 
            assert isinstance(hidden_layers, int)
            hidden_layers = [hidden_layers]
        
        self._validate_params(n_difference, direction_method)

        # initialize a DirectionFinder

        direction_finder = DIRECTION_FINDERS[direction_method](**direction_finder_kwargs)

		# if relevant, get the hidden state data for training set
        hidden_states = None
        
        n_tokens = [len(tokens) for tokens in self.tokenizer(train_inputs, padding=False)['input_ids']]

        if rep_token is None:
            # Convert case where rep_token is None to list[int] case          
            rep_token = len(train_inputs) * [-np.inf]
        elif isinstance(rep_token, list):
            # Convert positive token indices to negative indices
            rep_token = [-n_tokens[i] + rep_token[i] for i in range(len(rep_token))]
        
        hidden_states = None

        # Prevent converting to and from numpy arrays for methods where 
        # all the computation can be performed on the original device (i.e. GPU)
        SUPPORTED_METHODS = ["taylor", "least-squares"]
        if direction_method in SUPPORTED_METHODS:
            output_format = "torch"
            concat = torch.concatenate
        else:
            output_format = "numpy"
            concat = np.concatenate

        # If rep_token is list[int], average over the states for all
        # generated tokens after and including the token `rep_token[i]'.
        all_rep_tokens = [list(range(max(-n_tokens[i], rep_token[i]), 0)) \
            if isinstance(rep_token, list) else [rep_token] for i in range(len(train_inputs))]
        # Get the maximum number of tokens that the states for a train_input are averaged over
        # (e.g. if averaging over all tokens in a response, get the length of the longest response)
        len_tokens = [len(tokens) for tokens in all_rep_tokens]
        max_tokens = max(len_tokens)

        # Calculate the average of the hidden states across all required tokens for each train input,
        # or if average_rep_tokens is False, collect the activations.
        for i in range(len(train_inputs)):
            curr_inputs = train_inputs[i]
            curr_rep_tokens = all_rep_tokens[i]
            curr_hidden_states = None

            # Get the hidden states for each desired token in the current input
            for curr_rep_token in curr_rep_tokens:
                new_hidden_states = self._batched_string_to_hiddens(
                    [curr_inputs], curr_rep_token, hidden_layers, 1, which_hidden_states, output_format=output_format, **tokenizer_args
                )
                # Get cumulative sum of hidden states for the train input
                if average_rep_tokens:
                    if curr_hidden_states is None:
                        curr_hidden_states = new_hidden_states
                    else:
                        curr_hidden_states = {layer: state + new_hidden_states[layer] for layer, state in curr_hidden_states.items()}
                else:
                    if curr_hidden_states is None:
                        curr_hidden_states = {layer: state[..., None] for layer, state in new_hidden_states.items()}
                    else:
                        curr_hidden_states = {layer: concat((state, new_hidden_states[layer][..., None]), axis=2)\
                            for layer, state in curr_hidden_states.items()}
            
            # Convert sum to average
            if average_rep_tokens:
                curr_hidden_states = {layer: state / len(curr_rep_tokens) for layer, state in curr_hidden_states.items()}
            
            # When not averaging tokens, add padding to make all input
            # tensors have the same number of token dimensions
            else:
                for layer in curr_hidden_states.keys():
                    state = curr_hidden_states[layer]
                    shape = list(state.shape)
                    # Calculate padding length
                    shape[2] = max_tokens - shape[2]
                    padding = torch.zeros(shape, device=state.device, dtype=state.dtype)
                    curr_hidden_states[layer] = concat((state, padding), axis=2)
            
            # Group averaged hidden states for all train inputs       
            if hidden_states is None:
                hidden_states = curr_hidden_states
            else:
                hidden_states = {layer: concat((state, curr_hidden_states[layer]), axis=0) for layer, state in hidden_states.items()}

		# Run representation precomputation (i.e. pretraining)
        direction_finder.directions = direction_finder.get_rep_curves(
            hidden_states, hidden_layers,
            train_choices=train_labels,
            n_tokens=len_tokens
        )
        return direction_finder
    

class NonLinearRep(ABC):
    """Class to identify and store concept curves.
    
    Subclasses implement the abstract methods to identify concept curves 
    for each hidden layer via strategies including PCA, embedding vectors 
    (aka the logits method), and cluster means.

    Rep instances are used by RepPipeline to get concept scores.

    Curves can be used for downstream interventions."""

    @abstractmethod
    def __init__(self, n_components) -> None:
        colors = ['#e53d5c', '#d39f02', '#007050']
        colors2 = ['#e53d5cbb', '#e2bc4add', '#35b28fbb']
        self.cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)
        self.secondary_cmap = LinearSegmentedColormap.from_list("custom_cmap", colors2)
        self.n_components = n_components
        self.projectors = {}

    @abstractmethod
    def get_rep_curves(self, hidden_states, hidden_layers, **kwargs):
        """Get concept curves for each hidden layer of the model
        
        Args:
            hidden_states: Hidden states of the model on the training data (per layer)
            hidden_layers: Layers to consider

        Returns:
            curves: A dict mapping layers to curve arrays (n_components, hidden_size)
        """
        pass

    @abstractmethod
    def transform(self,
        hidden_states: dict[int, torch.Tensor],
        hidden_layers: list,
        *args, **kwargs) -> dict[int, np.ndarray]:
        """Project the hidden states onto the concept curves in self.curves

        Args:
            hidden_states: dictionary with entries of dimension (n_examples, hidden_size)
            hidden_layers: list of layers to consider

        Returns:
            transformed_hidden_states: dictionary with entries of dimension (n_examples,)
        """
        pass

    @abstractmethod
    def plot_representation(self, *args, **kwargs):
        """
        Plots the honesty representation as a new matplotlib figure,
        which can be saved, shown or added to tensorboard using plt.gcf().

        """
        pass

class CurveRep(NonLinearRep):
    """Get the curve that passes through the means for each honesty level."""
    
    def __init__(self, n_components: int = 2, poly_degree: int = 4, **_):
        """
        Args:
            poly_degree: Degree of polynomial to fit
        """
        super().__init__(n_components=n_components)
        self.poly_degree = poly_degree
        self.honesty_class_points = {}
        self.fit_params = {}
        
    def get_curve_points(self, x: np.ndarray | float, layer: int) -> np.ndarray:
        """
        Gets the points of the honesty curve for the other dimensions
        given the x (first principal component) values.

        Args:
            x
        Returns:
            curve_points:
                Axis 1: (*points_for_pcs_2+, honesty_points): Elements have the same type as x
        """
        scalar = False
        if not isinstance(x, np.ndarray):
            x = np.array(x).reshape(-1)
            scalar = True
        
        curve_points = [curve(x) for curve in self.fit_params[layer]]
        if scalar:
            # Convert to list of floats
            curve_points = [point.item() for point in curve_points]

        curve_points = np.array(curve_points).T
        return curve_points
    
    def plot_curve(self, X: np.ndarray, h: np.ndarray):
        """
        Plots the curve (i.e. for a single layer), coloured using the honesty scores
        Args:
            X (shape = [n_points, n_pca_dims]): Coordinates of the points
                on the curve (i.e. all principal components).
            h: Honesty values for each point in X
        Returns:
            line: For calling plt.colorbar(line) manually in case
                multiple curves will be plotted
        """
        # Convert points to line segments so that each segment can be coloured
        # Based on: https://matplotlib.org/stable/gallery/lines_bars_and_markers/multicolored_line.html#sphx-glr-gallery-lines-bars-and-markers-multicolored-line-py
        xy_points = X.reshape(-1, 1, X.shape[1])
        segments = np.concatenate([xy_points[:-1], xy_points[1:]], axis = 1)

        # Map h_curve (honesty values) to colours
        # h_colors = plt.Normalize(h_curve.min(), h_curve.max())
        if self.n_components >= 3:
            LineCollectFunc = Line3DCollection
            AddCollection = plt.gca().add_collection3d
        else:
            LineCollectFunc = LineCollection
            AddCollection = plt.gca().add_collection

        h_colors = plt.Normalize(-1, 1)

        color_line = LineCollectFunc(segments, cmap=self.cmap, norm=h_colors)
        color_line.set_array(h)
        color_line.set_linewidth(2)
        line = AddCollection(color_line)
        return line
    
    def transform(self,
        hidden_states: dict[int, torch.Tensor],
        hidden_layers: list,
        *args, **kwargs) -> dict[int, np.ndarray]:
        """
        Args:
            hidden_states: {layers: [n_examples, hidden_size]}
            hidden_layers: list of layers to consider
        Returns:
            honesty_values (shape = [n_examples])
        """
        honesty_values = {}

        for layer in hidden_layers:
            # Project hidden states onto principal components
            pca = self.projectors[layer]
            # shape = [n_examples, n_pcs]
            projected_state = pca.transform(hidden_states[layer].cpu().numpy())

            # The honesty score of the state is given by the honesty dimension of
            # the closest point on the curve
            def dist_from_point(point, x):
                """
                Distance between <point> and some point on the curve
                whose x-coordinate is <x>
                """
                curve_point = np.array([x, *self.get_curve_points(x, layer)[:-1]])
                return np.linalg.norm(point - curve_point)
            
            closest_x_coords = [minimize_scalar(partial(dist_from_point, point)).x for point in projected_state]
            honesty_coords = np.array([self.get_curve_points(x, layer)[-1] for x in closest_x_coords])
            honesty_values[layer] = honesty_coords

        return honesty_values

    def get_rep_curves(self,
        hidden_states: dict[int, np.ndarray],
        hidden_layers,
        **kwargs
    ) -> None:
        """
        Args:
            hidden_states
            
        Returns:
            None
        
        Attributes:
            self.honesty_class_points ({layer: {honesty_score: np.array [n_components]}})
            self.projectors ({layer: PCA component object})
            self.fit_params ({layer: [[PC2_curve_params], [PC3_curve_params], ... [honesty_curve_params]}):
                Using PC1 as the x-coordinates, these are the curve parameters for
                    calculating the other coordinates.
        """
        # train labels is necessary to differentiate between different classes
        train_choices = kwargs['train_choices'] if 'train_choices' in kwargs else None
        assert train_choices is not None, "CurveRep requires train_choices to differentiate two clusters"
        for layer in hidden_layers:
            assert len(train_choices) == len(hidden_states[layer]), f"Shape mismatch between hidden states ({len(hidden_states[layer])}) and labels ({len(train_choices)})"

        classes = sorted(list(set(train_choices)))
        n_hidden_dims = list(hidden_states.values())[0].shape[1]
        
        for layer in hidden_layers:
            """
            Get the principal components and the mean hidden state of each honesty
            class projected onto the principal components.
            """
            sum_hidden_states = {honesty_level: np.zeros((n_hidden_dims)) for honesty_level in classes}
            class_counts = {honesty_level: 0.0 for honesty_level in classes}
            
            # Compute sum of hidden states for each honesty class
            for sample_idx, row in enumerate(hidden_states[layer]):
                curr_class = train_choices[sample_idx]
                sum_hidden_states[curr_class] += row
                class_counts[curr_class] += 1

            # Elementwise divide along dimension 0 of sum_hidden_states
            avg_hidden_states = {hl: sum_hidden_states[hl] / class_counts[hl] for hl in classes}
        
            # PCA to convert avg_hidden_states from shape [n_classes x n_dims] to [n_classes x n_pca_dims]
            concatenated_avgs = np.array(list(avg_hidden_states.values()))
            pca = PCA(n_components=self.n_components).fit(concatenated_avgs)

            projected_hidden_states = {hl: pca.transform(state.reshape(1, -1)).squeeze().astype(np.float32) \
                for hl, state in avg_hidden_states.items()}
            self.projectors[layer] = pca
            self.honesty_class_points[layer] = projected_hidden_states

            """
            Fit the honesty curve to the honesty class points
            """

            # Put hidden states into a matrix with columns [PC 1, PC 2, ..., PC N, honesty value]
            points = np.array([
                np.concatenate((pcs.reshape(-1), [honesty])) for honesty, pcs in projected_hidden_states.items()
            ])
            x = points[:, 0]

            """Fit a curve to the honesty level points"""
            # Fit a curve for each dimension (i.e. honesty score and all PCs except first as each y-axis),
            # using the first PC as the x-axis
            fit_columns = list(range(1, self.n_components + 1))
            fit_params = [np.poly1d(np.polyfit(x, points[:, col], self.poly_degree)) for col in fit_columns]
            self.fit_params[layer] = fit_params

    def plot_representation(self, plot_layers: list = [-10]):
        """
        Args:
            plot_layers: Layers to plot
        """
        fig = plt.figure()

        if self.n_components >= 3:
            ax = plt.axes(projection='3d')
        
        for layer in plot_layers:
            """Plot points for each honesty class"""
            curve_points = self.honesty_class_points[layer]

            # Reshape dimensions of honesty points into columns
            points = np.array(list(curve_points.values()))
            # Ensure 2D
            points = points.reshape((points.shape[0], -1))
            # Plot at most 3 PC dimensions (+ colour for honesty score)
            points = [points[:, i] for i in range(min(points.shape[1], 3))]

            # Use the honesty score as the y-axis when using only 1 PCA dim
            if self.n_components == 1:
                points.append(np.array(list(curve_points.keys())))

            """Plot the curve that passes through the honesty points"""
            # Get points on the curve for plotting
            extrapolate_factor = 1.1
            x_curve = np.linspace(
                points[0].min() * extrapolate_factor,
                points[0].max() * extrapolate_factor
            )
            Y_curve = self.get_curve_points(x_curve, layer)
            h_curve = Y_curve[:, -1]

            # Always use colour for honesty scores, but only plot honesty scores in a spatial
            # dimension as well when there is 1 PC
            if self.n_components != 1:
                Y_curve = Y_curve[:, :-1]

            X_curve = np.concatenate((x_curve.reshape(-1, 1), Y_curve), axis=1)
            line = self.plot_curve(X_curve, h_curve)

            """Plot the points"""
            # Use the honest scores as the colour if there is only 1 layer
            scatter_color = {'c': list(curve_points.keys()), 'cmap': self.cmap} if len(plot_layers) == 1 else {'label': layer}
            if self.n_components >= 3:
                ax.scatter3D(*points, s=5, **scatter_color)
            else:
                plt.scatter(*points, s=6, **scatter_color)
            
        if len(plot_layers) != 1:    
            plt.legend(title="Layers", loc='upper left')
            title_layer = ""
        else:
            title_layer = f" for layer {plot_layers[0]}"

        # Honesty class is the mean activation of responses with same prompt prefix
        plt.title(f"Top {self.n_components} PCs of Honesty Representation{title_layer}")
        plt.xlabel("1st PC")
        plt.ylabel("2nd PC" if self.n_components > 1 else "Honesty Score")
        if self.n_components >= 3:
            ax.set_zlabel("3rd PC")
        
        # plt.xlim(-3, 4.5)
        #plt.ylim(-1.1, 3)

        plt.gca().set_aspect('equal', 'box')
        plt.colorbar(line)
        return fig
    
class Proj():

    def __init__(self):
        self.projector1 = None
        self.projector2 = None
        self.tensor_kwargs = dict(device="cuda", dtype=torch.float)
        
        # Number of dimensions for each linear regression model
        # (if using supervised projection for fit(), unless there are not enough input dimensions)
        self.linear_regression_dimensions = 2#5#100
        
        
    def test_fit(self, state: np.ndarray, nl_rep, labels: np.ndarray, **fit_kwargs):
        """
        Split the states into a training and test set to assess
        the generalisability of the linear model.
        """
        # Fit the projection to the training set
        data = np.concatenate((state, labels[..., None]), axis=1)
        train_set, test_set = train_test_split(data, train_size=0.7)
        self.fit(state=train_set[:, :-1], labels=train_set[:, -1], **fit_kwargs)
        
        # Calculate the accuracy metric:
        #   For each dimension, calculate the fraction of states where x_proj[i + 1] > x_proj[i]
        #   after sorting by target. That is, 100% accuracy occurs when the target concept score
        #   increases monotonically with the projection dimension. This metric is used as it
        #   is invariant to scaling.
        x_proj = self.transform(test_set[:, :-1])
        test_targets = test_set[:, -1]
        sorted_indices = np.argsort(test_targets)
        x_proj = x_proj[sorted_indices]
        score = np.zeros_like(x_proj[0])

        n_comparisons = x_proj.shape[0] - 1
        for i in range(n_comparisons):
            score += (x_proj[i + 1] >= x_proj[i]).astype(int)
        score /= n_comparisons

        for i, s in enumerate(score):
            print(f"Dimension {i} accuracy: {100*s.item():.1f}%")
        
        # Plot
        nl_rep.plot_representation(plot_layers=[], rep_layer=0)
        plt.gca().scatter(*[x_proj[:, i] for i in range(x_proj.shape[-1])], s=14, c=test_targets[sorted_indices], cmap=nl_rep.cmap)

        # Check projection using transform() and get_weights() are approximately equal
        w, b = self.get_weights()
        x_proj_original = self.transform(state)
        x_proj_manual = state[:, :self.max_input_dims] @ w + b
        projection_mse = ((x_proj_original - x_proj_manual)**2).mean()
        print(f"Manual projection MSE: {projection_mse}")

    def fit(self, state: np.ndarray, labels: np.ndarray, n_components: int, sparsity: float, rbf_kwargs, supervised_proj: bool=True):
        """
        Get the principal components for each concept class

        Args:
            state (shape=[n_samples, n_dims])
            labels (shape=[n_samples])
            n_components: Number of dimensions of the low dimensional space that the activations are projected onto.
        """
        self.max_input_dims = state.shape[-1]
        self.n_components = n_components
        pca_kwargs = dict(
            whiten=True,
            # Arpack is likely to be faster, but doesn't handle this case
            svd_solver="full" if min(state.shape) == n_components else "arpack"
        )
        two_stage = True
        if supervised_proj:
            self.n_dims = state.shape[1]
            self.dims_per_component = min(self.linear_regression_dimensions, (min(state.shape) - 1) // n_components)
            if self.dims_per_component == 0:
                raise ValueError(f"Require n_samples > n_components, but got n_samples = {state.shape[0]} and n_components = {n_components}")
            
            targets = np.expand_dims(labels, 1)
            if two_stage:
                self.projector1 = PCA(n_components=self.dims_per_component*n_components, **pca_kwargs).fit(state)
                # Project states onto an intermediate space using PCA
                regression_input = self.projector1.transform(state)
            else:
                regression_input = state
            targets = np.repeat(targets, repeats=n_components, axis=1)
            self.projector2 = [Ridge(alpha=0.1).fit(
                regression_input[:, start_dim::n_components],
                targets[:, start_dim::n_components]) for start_dim in range(n_components)
            ]

        elif sparsity == 0:
            self.projector1 = PCA(n_components=n_components, **pca_kwargs).fit(state)
            return self.projector1
        else:
            self.projector1 = SparsePCA(n_components=n_components, alpha=sparsity, max_iter=3).fit(state)
            return self.projector1

        return self
    
    def transform(self, x: np.ndarray) -> np.ndarray:
        """
        Project `x` onto `n_components` dimensions.

        Args:
            x (shape=[n_inputs, n_dims])
        Returns:
            x_proj (shape=[n_inputs, n_components] where n_inputs = n_concepts x n_variations_per_concept)
        """
        x_proj = x if self.projector1 is None else self.projector1.transform(x)
        if self.projector2 is not None:
            # Flatten and unflatten to ensure axis=1 exists
            x_proj = np.concatenate(
                [projector.predict(x_proj[:, dim::self.n_components]).flatten()[:, None] \
                    for dim, projector in enumerate(self.projector2)], 
                axis=1
            )
        return x_proj
    
    def get_weights(self):
        """
        For a projection in the form w @ x[..., :self.max_input_dims] - b, return (w, b).

        Returns:
            w (shape=[n_components, n_dims]): Projection weights
            b (shape=[n_components]): Bias to subtract after projection
        """
        x = np.random.randn(self.projector1.components_.shape[1])


        if self.projector1 is None:
            w, b = (None, None)
        elif isinstance(self.projector1, PCA) or isinstance(self.projector1, SparsePCA):
            # References: Based on sklearn.decomposition._base.py._BasePCA._transform
            # Get the weight and bias of the projection by performing the same
            # computations as sklearn.decomposition._base.py._BasePCA._transform
            # so that self.projector1.transform(x) == w1 @ x - b1
            # proj(x) = ((w @ x - w @ mean) / scale)
            w1 = self.projector1.components_
            mean = self.projector1.mean_
            b1 = w1 @ mean
            if self.projector1.whiten:
                scale = np.sqrt(self.projector1.explained_variance_)
                min_scale = np.finfo(scale.dtype).eps
                scale[scale < min_scale] = min_scale
                
                b1 = b1 / scale
                w1 = w1 / scale[..., None]

            w, b = (w1, b1)
        else:
            raise TypeError(f"Unexpected type for self.projector1: {type(self.projector1)}")

        if self.projector2 is not None:
            if isinstance(self.projector2, list):
                # Although the weights for each dimension are multiplied with alternating dimensions
                # of the input state `x`, which can be implemented by a dot product with the weights
                # for each dimension (i.e. w2[:, i] @ x[:, i:n_components:n_dims], 
                # where w2 has shape [n_components, n_dims//n_components]),
                # for compatibility, w2 is padded with alternating zeros such that w2 has shape 
                # [n_components, n_dims] and the only non-zero values of the ith component are
                # in the form i + k*n_components for int k.
                # This is so that doing w2 @ x[:, :n_dims] directly gives the same result as the dot product above.
                dims_per_component = self.projector2[0].coef_.shape[0]
                w2 = np.concatenate([
                    np.concatenate([
                        np.zeros((dims_per_component, component)),
                        self.projector2[component].coef_[:, None],
                        np.zeros((dims_per_component, self.n_components - component - 1))
                    ], axis=1).flatten()[None, :]
                    for component in range(self.n_components)
                ], axis=0)
                # Concatenate the linear model weights for each projection dimension
                get_attr = lambda attr: np.concatenate([attr(model)[:, None] for model in self.projector2], axis=1)
                b2 = np.concatenate([model.intercept_ for model in self.projector2])#get_attr(lambda model: model.intercept_)
            elif isinstance(self.projector2, PLSRegression):
                w2 = self.projector2.coef_
                b2 = (self.projector2._x_mean @ w2) - self.projector2.intercept_
            else:
                raise TypeError(f"Unexpected type for self.projector2: {type(self.projector2)}")
            
            w, b = (w2, b2) if w is None else (w2 @ w, w2 @ b + b2)

        return w, b
    
    def get_direction_vector(self):
        """Get a direction vector from the projection weights to use for linear control."""
        w, _ = self.get_weights()
        return w if len(w.shape) == 1 else w.mean(axis=1)

def save_projection(projection):
    try:
        with open("projection.pickle", "wb") as f:
            pickle.dump(projection, f)
    except Exception as e:
        print(f"Error: {e}")

def load_projection():
    """To measure the similarity between projection weights"""
    try:
        with open("projection.pickle", "rb") as f:
            projection = pickle.load(f)
    except Exception as e:
        print(f"Error: {e}")
    return projection


class SurfaceRep(NonLinearRep):
    def __init__(self,
            n_components: int = 2,
            sparsity: float = 0.0,
            rbf_kwargs: dict = DEFAULT_RBF_KWARGS,
            **_
        ):
        """
        Args:
            sparsity: Increasing sparsity reduces computation time for control, but increases
                the number of hidden state dimensions that are not controlled.
        """
        super().__init__(n_components=n_components)
        self.rbf_kwargs = rbf_kwargs
        self.sparsity = sparsity
        self.means = {}
        self.stds = {}

        # projected_states[layer] is a numpy array with shape [n_samples, n_components]
        self.projected_states = {}
        # Honesty values for samples in self.projected_states. List with length n_samples
        # All layers have the same values
        self.honesty_values = None
        self.surface = {}

        if n_components < 2:
            raise ValueError(f"SurfaceRep requires at least 2 principal components, but got n_components = {n_components}")
        
    def plot_representation(self,
        plot_layers: list = list(range(-10, -25, -1)),
        rep_layer: int = -10,
        n_axes: int = 2
    ):
        """
        Args:
            plot_layers: Plot the projection of points for all these layers
            rep_layer: Plot the rep surface for this layer and plot large points for this layer.
                Must be in plot_layers
            n_axes: Number of axes to use for the plot. Should match
                plot_func.
        """
        lim = 1.5
        plt.figure(figsize=(7, 6), dpi=400)
        ax = plt.axes(projection='3d') if n_axes == 3 else plt.axes()
        # if rep_layer not in plot_layers:
        #     raise ValueError(f"rep_layer '{rep_layer}'must be in plot_layers")
        color_scale = None

        n_grid_points = 100
        zero_grids = [np.zeros(shape=(n_grid_points**2, 1)) for _ in range(self.n_components - 2)]


        # Convert points to grid
        grid_points = np.linspace(-lim, lim, n_grid_points)
        x_grid, y_grid = np.meshgrid(grid_points, grid_points)
        concept_grid = self.surface[rep_layer](
            np.concatenate((x_grid.reshape((-1, 1)), y_grid.reshape((-1, 1)), *zero_grids), axis=1)
        ).reshape(x_grid.shape)
        if n_axes == 3:
            ax.plot_surface(x_grid, y_grid, concept_grid, cmap=self.secondary_cmap, alpha=0.5)
        else:
            clipped_concept_grid = np.clip(concept_grid, a_min=-1, a_max=1)
            surface_color_scale = plt.contourf(x_grid, y_grid, clipped_concept_grid, cmap=self.secondary_cmap)
        # TODO: Make plt.contour levels the sames as the concept levels used to fit the representation and not arbitrary
        # Plot contours at regular concept level intervals
        plt.contour(x_grid, y_grid, concept_grid, colors="black", linewidths=0.5, levels=list(np.arange(-1, 1.01, 0.2)))
        
        
        # Plot all points in required layers
        for layer in plot_layers:
            points = self.projected_states[layer]

            # Ensure 2D and separate columns
            points = [points[:, i] for i in range(2)]

            # Plot the points
            # Use the concept scores as the colour if there is only 1 layer
            point_size = 14 if layer == rep_layer else 1
            scatter_color = {'c': self.honesty_values, 'cmap': self.cmap}
            if n_axes >= 3:
                point_color_scale = ax.scatter(*points, self.honesty_values, s=point_size - 2, **scatter_color)
            else:
                point_color_scale = ax.scatter(*points, s=point_size, **scatter_color)
    
    
        # Honesty class is the mean activation of responses with same prompt prefix
        # plt.title(f"Top {self.n_components} dimensions of Concept Representation")
        plt.xlabel("1st dimension")
        plt.ylabel("2nd dimension")
        if n_axes >= 3:
            ax.set_zlabel("Concept Score")
            ax.set_zlim(-1, 1)
        
        plt.xlim(-lim, lim)
        plt.ylim(-lim, lim)

        # ax.set_aspect('equal', 'box')
        if rep_layer in plot_layers:
            if n_axes < 3:
                cbar1 = plt.colorbar(point_color_scale, aspect=50, pad=-0.12)
                cbar2 = plt.colorbar(surface_color_scale, aspect=30, pad=0.01)
                ticks = cbar2.get_ticks()
                cbar1.set_ticks(ticks=ticks, labels=[round(t, 1) for t in ticks])
                cbar2.set_ticks(ticks=ticks, labels=['']*len(ticks))
            
        plt.tight_layout()
    
    def get_rep_curves(self,
            hidden_states: dict[int, np.ndarray],
            hidden_layers,
            **kwargs
        ) -> None:
        """
        Args:
            hidden_states: Dict in the form
                {layer_idx: states (shape=[n_concepts * n_variations, n_hidden_dims])}
        """
        # train labels is necessary to differentiate between different classes
        train_choices = kwargs['train_choices'] if 'train_choices' in kwargs else None
        assert train_choices is not None, "SurfaceRep requires train_choices to differentiate two clusters"
        for layer in hidden_layers:
            assert len(train_choices) == len(hidden_states[layer]), f"Shape mismatch between hidden states ({len(hidden_states[layer])}) and labels ({len(train_choices)})"

        self.honesty_values = train_choices
        fit_proj = lambda **kwargs: Proj().fit(**kwargs)
        fit_proj_kwargs = dict(
            labels=np.array(train_choices),
            n_components=self.n_components,
            sparsity=self.sparsity,
            rbf_kwargs=self.rbf_kwargs
        )
        if min(next(iter(hidden_states.values())).shape) < self.n_components:
            raise ValueError("n_components must be less than the number of example prompts "
                "(e.g. prompt templates) used to train the representation.")
        
        ###DEBUG: Disable when not debugging
        self.fit_proj_kwargs = fit_proj_kwargs
        self.states = hidden_states[hidden_layers[0]]
        ###

        n_procs = 1 if self.sparsity == 0 else min(psutil.cpu_count(logical=False), len(hidden_layers))
        # Since there is only one process required, avoid the overhead of creating a process
        # (which may be substantial if hidden_states is large)
        if n_procs == 1:
            for layer in hidden_layers:
                # Normalise hidden states before computing PCs to ensure different layers
                # have similar numbers of non-zero PCs
                state = hidden_states[layer]

                projector = fit_proj(state=state, **fit_proj_kwargs)
                self.get_rep_curves_fit_representation_(projector, state, layer, train_choices)
        
        else:
            results = {}
            curr_state = {}
            with mp.Pool(processes=n_procs) as pool:
                for layer in hidden_layers:
                    # Normalise hidden states before computing PCs to ensure different layers
                    # have similar numbers of non-zero PCs
                    state = hidden_states[layer]
                    ##self.means[layer] = np.mean(state, axis=0)
                    ##state -= self.means[layer]
                    
                    # Scale by std across all dimensions rather than scaling each dimension
                    # by the dimension's std. Otherwise unimportant dimensions with low std 
                    # may be 'selected' (i.e. non-zero PC) by SparsePCA
                    ##stds = 5.0 * np.std(state)
                    # Prevent division by 0 when scaling
                    ##self.stds[layer] = 0.1 if np.abs(stds) < 0.1 else stds
                    ##state /= self.stds[layer]
                    curr_state[layer] = state

                    results[layer] = pool.apply_async(fit_proj, kwds=dict(state=curr_state[layer], **fit_proj_kwargs))
                for layer in hidden_layers:
                    projector = results[layer].get()
                    self.get_rep_curves_fit_representation_(projector, curr_state[layer], layer, train_choices)

    def get_rep_curves_fit_representation_(self, projector, layer_state, layer, train_choices):
        """
        Helper for self.get_rep_curves().

        Project the states for the current layer onto the principal component
        space and extract the surface representation.
        """
        proj_states = projector.transform(layer_state)
        
        # Fit a surface to the points
        rbf = RBFInterpolator(
            proj_states,
            train_choices,
            **self.rbf_kwargs
        )
        self.projectors[layer] = projector
        self.projected_states[layer] = proj_states
        self.surface[layer] = rbf

    def transform(self,
        hidden_states: dict[int, torch.Tensor],
        hidden_layers: list,
        *args, **kwargs) -> dict[int, np.ndarray]:
        """
        Classify concept score of hidden states

        Args:
            hidden_states: {layers: [n_examples, hidden_size]}
            hidden_layers: list of layers to consider
        Returns:
            honesty_values (shape = [n_examples])
        """
        honesty_values = {}

        for layer in hidden_layers:
            # Project hidden states onto principal components
            projector = self.projectors[layer]
            # shape = [n_examples, n_pcs]
            ##curr_state = (hidden_states[layer].cpu().numpy() - self.means[layer]) / self.stds[layer]
            curr_state = hidden_states[layer].cpu().numpy()
            projected_state = projector.transform(curr_state)
            honesty_values[layer] = self.surface[layer](projected_state)
        return honesty_values

    def control_layer(self,
        hidden_states: torch.Tensor,
        layer: int,
        new_score: float = 3.0,
        is_score_relative: bool = False,
        max_offset: float = None,
        use_same_bounds: bool = True,
        control_last_token: bool = True,
        solver: str = "lstsq",
        layer_weights: dict | None = None,
        all_layer_scale: float = 0.25
    ) -> torch.Tensor:
        """
        Control the hidden state such that the projection
        of the new hidden state has a new concept score as close
        as possible to <new_score>
        Args:
            hidden_states (shape=[batch_size, ...]): Embeddings for the current layer
            new_score: Desired concept score.
            is_score_relative:
                If True, new_score is the desired change in score
                If False, new_score is the desired absolute concept score
                E.g. if the concept score of the initial state = 0.3 and new_score = -1,
                    the controlled state will have score -0.7 and -1 for True and False respectively.
            max_offset: 
                If use_same_bounds is True, this is the global offset (relative to the origin).
                    Defaults to 5 (i.e. virtually unbounded)
                If use_same_bounds is False, this is the maximum distance from the point with 
                the desired concept strength in the low-dimensional space that is considered acceptable.
                    Defaults to 0.5.
            use_same_bounds: If False, bounds for calculating optimal concept scores are relative
                to each prompt, so the optimization needs to be computed for each prompt.
                If True, save computation by using the same (global) bounds for all prompts.
            control_last_token: If True, control only the last token (i.e. the new predicted token).
                This assumes that the second dimension is the number of tokens. Typically
                shape=[n_prompts, n_tokens, dim_hidden_state]. Otherwise, shape
                of hidden_states can be arbitrary.
            solver: Method to find the controlled hidden state given the PC projection of that state.
                Choices:
                    "lstsq": Use a least-squares solver. Solution that minimises the L2-norm,
                        but slow and doesn't work for large arrays
                    "analytic_l2": Analytic solution
            layer_weights: If not None, multiply the change in state by layer_weights[layer].
                Dict of (layer_index, weight) pairs.
            all_layer_scale: Constant weight applied to all layers
        Returns:
            hidden_states with the last token controlled                                          
        """


        projector = self.projectors[layer]
        get_score = lambda proj_state: self.surface[layer](proj_state).item()
        batch_size = hidden_states.shape[0]

        ### Get initial values
        # Calculate a new hidden state that is close to the initial
        # state (minimize), but which maps to the desired projected state (equality constraint)
        if control_last_token:
            controlled_state = hidden_states.detach().clone()
            initial_state = controlled_state[:, -1, :]
        else:
            initial_state = hidden_states

        initial_state = initial_state.float().cpu().numpy().astype(np.float32)
        initial_proj = projector.transform(initial_state) 

        desired_proj = np.zeros_like(initial_proj)

        ### Set default arguments
        if solver == "lstsq" and projector.n_dims > 1e8:
            warnings.warn("Least squares solver does not support large array sizes. Resorting to 'analytic_l2'.")
            solver = "analytic_l2"

        old_score = new_score
        if is_score_relative:
            # Convert relative to absolute score
            initial_score = get_score(initial_proj)
            new_score = initial_score + new_score

        if max_offset is None:
            max_offset = 5.0 if use_same_bounds else 0.5
       

        #Ablate surface and measure control ability without it
        if ABLATE_NON_LINEAR_CONTROL: 
            direction_vector = projector.get_direction_vector()#W = projector.projector.linear.weight[0, :]#.T @ projector.projector.linear2.weight
            direction_unit_vector = direction_vector / np.linalg.norm(direction_vector)
            last_token_controlled_state = initial_state
            last_token_controlled_state[:, :projector.max_input_dims] += old_score * direction_unit_vector#.unsqueeze(0).cpu().detach().numpy()
        else:
            # Find the point with the desired concept score, using a small regularisation term to minimise the distance
            # of the new projected point from the initial projected point, so that the change in state required to produce
            # the desired change in projection is less.
            loss_function = lambda point: (get_score(point[None, ...]) - new_score)**2 + 1e-4*np.linalg.norm(point[None, ...] - initial_proj)
            
            ### Get optimal projected state
            if use_same_bounds:
                bounds = [(-max_offset, max_offset) for _ in range(self.n_components)]
                # Ensure initial_proj lies within the bounds
                initial_proj = np.clip(initial_proj, -max_offset, max_offset)
                
                # Calculate and collect desired projections for each batch into an np.ndarray
                desired_proj = [differential_evolution(
                    func=loss_function,
                    x0=initial_proj[batch, ...],#np.zeros((self.n_components,)),#
                    atol=1e-6,#0.0005,
                    bounds=bounds
                ).x[None, ...] for batch in range(batch_size)]
                desired_proj = np.concatenate(desired_proj, axis=0)
            else:
                raise ValueError("use_same_bounds == False is deprecated")


            delta_proj = desired_proj - initial_proj

            ### Get the controlled, high-dimensional state given the desired projection
            # Constraint: ax ~= b (i.e. proj(delta_state) ~= delta_proj)
            # Solve a set of linear equations (#hidden_dims variables with #PC equations)
            # Undo normalization (multiply by std) from projector.transform() by multiplying b
            w_projection, b_projection = projector.get_weights()
            if solver == "lstsq":
                delta_state = scipy.linalg.lstsq(
                    a=w_projection,
                    b=delta_proj.T
                )[0].T
            elif solver == "analytic_l2":
                norms = np.linalg.norm(w_projection, axis=1)
                coefficients = delta_proj / norms
                delta_state = coefficients @ w_projection

            else:
                raise ValueError(f"Unsupported solver '{solver}'")

            weight = 1.0 if layer_weights is None else layer_weights[layer]

            # No need to add mean to undo normalization since it is included in the unnormalized initial_state
            scaled_delta_state = delta_state * weight
            #norm = np.linalg.norm(scaled_delta_state)
            scaled_delta_state *= all_layer_scale#3/len(range(-5, -26, -1))#max(3*np.sqrt(norm), 1)
            last_token_controlled_state = scaled_delta_state + initial_state

        # Update hidden state with controlled values
        new_state = torch.from_numpy(last_token_controlled_state).to(dtype=hidden_states.dtype, device=hidden_states.device)

        if control_last_token:
            controlled_state[:, -1, :] = new_state
        else:
            controlled_state = new_state

        return controlled_state

class TaylorRep():
    """
    A local representation of concepts using a Taylor approximation and
    approximating derivatives using central finite differences.
    """
    def __init__(self, order: int, **_):
        """
        Args:
            order: Order of the Taylor polynomial
        Usage note:
            To use this class via the configs in eval_config.py,
            pass rep_reader_kwargs={"order": <ORDER>} to RepConfig
            and order=<order> to RepInferenceConfig,
            to pass `order` here and construct the corresponding
            prompt template, respectively.
        """
        self.N = order
        self.PRECISION = 5

    def order_to_concept_levels(order: int):
        """
        Gets the concept levels required to compute the 
        coefficients of the Taylor polynomial of the given order.
        E.g. for order = 2, concept_levels = [-1, -0.5, 0, 0.5, 1]
        """
        concept_levels = list(np.linspace(-1, 1, 2*order + 1))
        #concept_levels.remove(0)
        return concept_levels

    def get_rep_curves(self,
            hidden_states: dict[int, torch.Tensor],
            hidden_layers,
            **kwargs
        ) -> None:
        """
        Preprocessing needed to compute the representation in the form of the coefficients
        of the taylor polynomial. The coefficients are computed in control_layers since the
        coefficients of the terms with even degree depend on states with a neutral concept
        level (which the uncontrolled response is use for).

        Args:
            hidden_states: Dict in the form
                {layer_idx: states (shape=[n_concepts * n_variations, n_hidden_dims])}
            hidden_layers: Keys of hidden_states
        Attributes:
            self.polynomial_coefs (shape=[order, n_dims]): 
                self.polynomial_coefs[i] is the coefficient of 
                (x - x0)**(i + 1) in the Taylor polynomial.
            self.layers: hidden_layers
            self.collect_states: States used to fit the Taylor polynomial
        """
        self.layers = hidden_layers
        # Train labels is necessary to differentiate between different classes
        train_choices = kwargs['train_choices'] if 'train_choices' in kwargs else None
        assert train_choices is not None, "TaylorRep requires train_choices to differentiate two clusters"
        for layer in hidden_layers:
            assert len(train_choices) == len(hidden_states[layer]), f"Shape mismatch between hidden states ({len(hidden_states[layer])}) and labels ({len(train_choices)})"

        # Assumes that the variations for the same concept level are consecutive in train_choices, but
        # concept levels are not necessarily sorted 
        concept_levels = list(set(train_choices))
        n_concepts = len(concept_levels)
        n_variations = len(train_choices) // n_concepts
        concept_levels = train_choices[::n_variations]

        self.honesty_values = concept_levels
        first_state = hidden_states[hidden_layers[0]]
        shape = list(first_state.shape)
        n_dims = shape[1]
        device = first_state.device

        tensor_kwargs = dict(dtype=torch.float32, device=device)
        self.polynomial_coefs = torch.zeros([self.N, n_dims], **tensor_kwargs)

        # Reshape states in hidden_states to [n_concepts, n_variations, n_hidden_dims] and
        # compute the mean of the variations for each concept level (result shape=[n_layers, n_concepts, n_dims])
        hidden_states = torch.cat([torch.mean(states.reshape(n_concepts, n_variations, *shape[1:]),  dim=1)[None, ...] for states in hidden_states.values()], dim=0)
        # Convert hidden_states to the shape {n_concepts: [n_layers, n_dims]}
        self.collected_states = {round(concept_level, self.PRECISION): hidden_states[:, i, :].to(**tensor_kwargs) for i, concept_level in enumerate(concept_levels)}


    def control_layer(self,
        hidden_states: torch.Tensor,
        layer: int,
        new_score: float = 3.0,
        **_
    ) -> torch.Tensor:
        """
        Control the hidden states for the last token of layer `layer`.
        This is given by taylor_polynomial(new_score)

        Args:
            hidden_states (shape=[1, n_tokens, n_dims] for 
                NLP or [1, 1, flatten_dims] for image generation)
        """
        x0 = hidden_states.detach().clone()[:, -1, :]
        # Ensure tensors are all on the same device
        tensor_kwargs = dict(device=x0.device, dtype=torch.float32)
        x0 = x0.to(**tensor_kwargs)
        self.polynomial_coefs = self.polynomial_coefs.to(**tensor_kwargs)
        layer_idx = self.layers.index(layer)

        #######################################################
        # Calculate coefficients for each degree (n) of the polynomial
        #######################################################
        h = 2 / self.N
        for n in range(1, self.N + 1):
            
            coef = torch.zeros_like(self.polynomial_coefs[n - 1])

            for i in range(0, n + 1):
                # Get the term x_0 + (n/2 - i)h where x_0 = 0
                # Round to precision to prevent potential key errors due to floating point errors
                concept_score = round((n/2 - i) * h, self.PRECISION)
                scalar = ((-1) ** i) * math.comb(n, i)

                # Unexpected concept score    
                if concept_score not in self.collected_states.keys():
                    required_concept_levels = TaylorRep.order_to_concept_levels(self.N)
                    raise ValueError(f"Concept levels must be {required_concept_levels} for "\
                        f"Taylor series order {self.N}, but got concept_levels={self.honesty_values}")
                # Contribution from prompting a model to act non-neutral
                else:
                    coef += scalar * self.collected_states[concept_score][layer_idx].to(**tensor_kwargs)

            self.polynomial_coefs[n - 1] = coef / (math.factorial(n) * (h**n))


        #######################################################
        # Calculate the controlled state
        #######################################################
        

        # Ensure tensors are on the same device
        D = self.polynomial_coefs.to(device=x0.device)
        controlled_state = x0
        c = new_score

        for n in range(1, self.N + 1):
            controlled_state += (c**n) * D[n - 1]

        hidden_states[:, -1, :] = controlled_state
        return hidden_states

class LstsqRep():
    """
    A local representation of concepts by fitting a polynomial
    approximation through the averaged activations for each concept level.
    """
    def __init__(self, order: int, **_):
        """
        Args:
            order: Order of the Taylor polynomial
        Usage note:
            To use this class via the configs in eval_config.py,
            pass rep_reader_kwargs={"order": <ORDER>} to RepConfig
            and order=<order> to RepInferenceConfig,
            to pass `order` here and construct the corresponding
            prompt template, respectively.
        """
        self.N = order
        self.PRECISION = 5

    def get_rep_curves(self,
            hidden_states: dict[int, torch.Tensor],
            hidden_layers,
            n_tokens: list[int],
            **kwargs
        ) -> None:
        """
        Preprocessing needed to compute the representation in the form of the coefficients
        of the polynomial.

        Args:
            hidden_states: Dict in the form
                {layer_idx: states (shape=[n_concepts * n_variations, n_hidden_dims, Optional[n_tokens]])}
            hidden_layers: Keys of hidden_states
            n_tokens: Number of tokens for each value in hidden_states. Used to determine the amount
                of padding if hidden_states has states for each token (has a tokens axis).
                hidden_states[layer][i, :, n_tokens[i]:] is padded with zeros for all layers.
        Attributes:
            self.layers: hidden_layers
            self.collect_states (shape={n_concepts: [n_layers, n_dims, Optional[n_tokens]]}): States used to fit the polynomial
            self.n_tokens: n_tokens
            self.current_token: How many tokens of the response have been controlled so far. Initially set to 0 if hidden_states has a
            tokens axis (i.e., 3rd axis), otherwise set to None if the same representation is used for controlling all tokens.
        """
        self.layers = hidden_layers
        self.n_tokens = n_tokens
        # Train labels is necessary to differentiate between different classes
        train_choices = kwargs['train_choices'] if 'train_choices' in kwargs else None
        assert train_choices is not None, "LstsqRep requires train_choices to differentiate two clusters"
        for layer in hidden_layers:
            assert len(train_choices) == len(hidden_states[layer]), f"Shape mismatch between hidden states ({len(hidden_states[layer])}) and labels ({len(train_choices)})"

        # Assumes that the variations for the same concept level are consecutive in train_choices, but
        # concept levels are not necessarily sorted 
        concept_levels = list(set(train_choices))
        n_concepts = len(concept_levels)
        n_variations = len(train_choices) // n_concepts
        concept_levels = train_choices[::n_variations]

        self.honesty_values = concept_levels
        first_state = hidden_states[hidden_layers[0]]
        shape = list(first_state.shape)
        self.n_dims = shape[1]
        device = first_state.device

        if self.N > len(concept_levels) + 1:
            raise ValueError(f"Order of the polynomial ({self.N}) is greater than the number of concept levels ({len(concept_levels) + 1})")

        tensor_kwargs = dict(dtype=torch.float32, device=device)

        # Reshape states in hidden_states to [n_concepts, n_variations, n_hidden_dims, Optional[n_tokens]] and
        # compute the mean of the variations for each concept level (result shape=[n_layers, n_concepts, n_dims, Optional[n_tokens]])
        hidden_states = torch.cat([torch.mean(states.reshape(n_concepts, n_variations, *shape[1:]),  dim=1)[None, ...] for states in hidden_states.values()], dim=0)
        # Convert hidden_states to the shape {n_concepts: [n_layers, n_dims, Optional[n_tokens]]}
        self.collected_states = {concept_level: hidden_states[:, i, :].to(**tensor_kwargs) for i, concept_level in enumerate(concept_levels)}
        self.current_token = None if len(shape) == 2 else 0

    def control_layer(self,
        hidden_states: torch.Tensor,
        layer: int,
        new_score: float = 3.0,
        **_
    ) -> torch.Tensor:
        """
        Control the hidden states for the last token of layer `layer`.
        This is given by lstsq_polynomial(new_score)

        Args:
            hidden_states (shape=[1, n_tokens, n_dims])
            layer: The layer relative to the end of the model
                (e.g. -2 is the second last layer)
        """
        layer_idx = self.layers.index(layer)
        x0 = hidden_states.detach().clone()[0, -1, :]
        # Ensure tensors are all on the same device
        tensor_kwargs = dict(device=x0.device, dtype=torch.float32)
        x0 = x0.to(**tensor_kwargs)
        
        #######################################################
        # Calculate coefficients of the polynomial and control the state
        #######################################################
        # Extract x (list of concept levels with shape=[n_concepts]) and
        # y (Tensor of activations for the current layer for each concept level, 
        # with shape=[n_dims, n_concepts]) from collect_states
        x = torch.tensor(list(self.collected_states.keys()), **tensor_kwargs)
        y = torch.cat([state[layer_idx].unsqueeze(1) for state in list(self.collected_states.values())], dim=1).to(**tensor_kwargs)
        
        if self.current_token is not None:
            # Get the indices of the concept levels where the response has at least `self.current_token'
            usable_concept_levels = torch.where(torch.tensor([tokens_for_concept_level > self.current_token for tokens_for_concept_level in self.n_tokens]))[0]
            if len(usable_concept_levels) < 2:
                return hidden_states
            x = x[usable_concept_levels]
            y = y[:, usable_concept_levels, self.current_token]
            self.current_token += 1

        # Convert x to a tensor 'a' with dim 2 in the form [1, X, X^2, ...].
        # a has shape [d, |L|, N + 1]. Least squares is compute for a batch of d, where d is the number of dims/layer.
        # polynomials (d is effectively the batch size in lstsq).
        a = torch.ones_like(x)
        a = a.unsqueeze(dim=1)

        for _ in range(self.N):
            next_column = (a[:, -1] * x).squeeze().unsqueeze(dim=1)
            a = torch.cat((a, next_column), dim=1)

        a = a.unsqueeze(dim=0)
        a = a.repeat((y.shape[0], 1, 1))

        # Get weights (shape=[n_dims, self.N + 1])
        weights = torch.linalg.lstsq(a, y).solution

        # Compute poly(new_score) - weights[:, 0] + x0 to shift the polynomial
        # to be centred around x0
        controlled_state = x0.clone().detach()
        for degree in range(1, self.N + 1):
            controlled_state += (new_score ** degree) * weights[:, degree]


        hidden_states[0, -1, :] = controlled_state
        return hidden_states


DIRECTION_FINDERS = {
    'curve': CurveRep,
    'surface': SurfaceRep,
    'taylor': TaylorRep,
    'least-squares': LstsqRep
}