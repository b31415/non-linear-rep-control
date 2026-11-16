"""
Gradio UI for playing around with each representation engineering method.
"""
import io
import json
import gradio as gr
from nltk.corpus import wordnet as wn
import numpy as np
import matplotlib.pyplot as plt
from transformers import pipeline

from examples.honesty.rep_manager_utils import RepManager, load_model
from evaluate.helpers.eval_init_model import load_rep_method
from evaluate.eval_config import RepConfig, RepInferenceConfig, BenchmarkMetadata, LAYER_WEIGHTS
from repe import rep_utils

def classify_response(
    concept: str,
    concept_antonym: str,
    concept_description: str,
    prompt: str,
    response: str,
    plot_layer: int,
    method: str
) -> tuple[gr.Label, gr.Image, gr.Label]:
    # Format prompt
    prompt_template = {"role": "user", "content": prompt}
    response_template = [prompt_template, {"role": "assistant", "content": response}]
    prompt_response = rep_utils.apply_chat_template(response_template, tokenizer=tokenizer)

    if method in CLASSIFICATION_NOT_IMPLEMENTED:
        # Representation plotting and classification is not implemented
        # for "Contrast Vector".
        return None, None
    
    # Get the length of the prompt without the start of the response
    # since we are using the representation to classify a user-given or uncontrolled response
    prompt_len = len(tokenizer.apply_chat_template([prompt_template], tokenize=False))
    
    # Fit representation
    inference_conf = RepInferenceConfig(
        benchmark=BenchmarkMetadata(
            concept_words=(concept, concept_antonym, concept_description)      
        ),
        representation_method=conf.representation_method,
        control_strength=control_strength,
        order=POLYNOMIAL_ORDER,
        control_func_kwargs={"all_layer_scale": 1/3},
        **REP_INFERENCE_KWARGS
    )
    rep_manager.train_rep_reader_on_prompt([prompt_template], **inference_conf.train_rep_reader_kwargs)
    
    # Classify response
    response_classification = rep_manager.classify_tokens(
        [prompt_response], response_start_pos=prompt_len
    )[0]

    # Get classification label
    label = "Overall classification score: "
    if response_classification > 0.3:
        label += concept
    elif response_classification < -0.3:
        label += concept_antonym
    else:
        label += "Uncertain"

    return gr.Label(round(response_classification, 3), label=label), plot(plot_layer, method)

def control_response(
    concept: str,
    concept_antonym: str,
    concept_description: str,
    prompt: str,
    response: str,
    control_strength: float
) -> tuple[gr.Label, gr.Image, gr.Label]:
    
    print(50*"=")
    print("Controlling Response")
    # Format prompt
    response_template = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response}
    ]

    # Use absolute scoring for demo
    # is_score_relative = False for all SurfaceRep qualitative results, unless specified
    control_func_kwargs = {"is_score_relative": True, "layer_weights": LAYER_WEIGHTS}

    # Control
    inference_conf = RepInferenceConfig(
        benchmark=BenchmarkMetadata(
            concept_words=(concept, concept_antonym, concept_description)      
        ),
        representation_method=conf.representation_method,
        control_strength=control_strength,
        order=POLYNOMIAL_ORDER,
        control_func_kwargs=control_func_kwargs,
        **REP_INFERENCE_KWARGS
    )


    controlled_response = model.generate(
        response_template,
        control_strength=control_strength,
        train_rep_reader_kwargs=inference_conf.train_rep_reader_kwargs,
        base_model_forward_kwargs=base_model_forward_kwargs,
        control_func_kwargs=inference_conf.control_func_kwargs
    )["output"]
    controlled_response = tokenizer.batch_decode(controlled_response, skip_special_tokens=True)[0]

    return controlled_response

def plot(plot_layer, method):
    # Representation not initialised yet
    if rep_manager.rep_reader is None:
        return None
    
    # Method specific args
    if method.lower() == 'surface':
        plot_args = {"plot_layers": rep_manager.hidden_layers, "rep_layer": plot_layer}
    else:
        plot_args = {"plot_layers": [plot_layer]}

    rep_manager.rep_reader.plot_representation(**plot_args)
    
    fig = plt.gcf()
    fig.set_figheight(5)
    fig.set_dpi(400)

    # Save plot to numpy array
    io_buf = io.BytesIO()
    fig.savefig(io_buf, format='raw', dpi=400)
    io_buf.seek(0)
    image = np.reshape(
        np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
        newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
    io_buf.close()
    return gr.Image(image, label="Layer -10 concept representation")

def generate_response(prompt: str) -> str:
    return pipe(prompt, return_full_text=False, **base_model_forward_kwargs)[0]['generated_text'].strip()

def dropdown_load_method(method: str) -> tuple[gr.Column, gr.Button]:
    """
    Load the new representation method on dropdown change and update
    the visibility of components that should be hidden if the method
    does not support classification.

    Returns:
        Column: Contains classification label and representation plot
        Classification button
    """
    # Can't pass model or rep_manager as arguments using i.e. gr.State(value=model)
    # as this will deepcopy the model and use unnecessary
    global model, rep_manager, conf
    print(f"Loading method: {method}")
    conf.representation_method = method
    model, _, rep_manager = load_rep_method(conf, model=base_model, tokenizer=tokenizer)
    enable_classification = (method not in CLASSIFICATION_NOT_IMPLEMENTED)

    # Create updated components
    classification_display = gr.Column(visible=enable_classification)
    classify_button = gr.Button(visible=enable_classification, **classify_kwargs)
    return classification_display, classify_button

def fill_preset_values(preset: str) -> dict:
    """Get values for each input textbox for the selected preset"""
    # Use eval to convert from string to variable (e.g. "concept" to concept)
    return {eval(variable_name): value for variable_name, value in presets[preset].items()}

base_model_forward_kwargs = dict(max_new_tokens=40)
base_model, tokenizer = load_model("mistralai/Mistral-7B-Instruct-v0.2", compile=False, flash_attn=False)

pipe = pipeline(task="text-generation", model=base_model, tokenizer=tokenizer)
# Layers to calculate the representation for and plot
hidden_layers = (-5, -25, -1)#(-1, -32, -2)

prev_method = None
POLYNOMIAL_ORDER = 5
rep_reader_kwargs = {"n_components": 2, "order": POLYNOMIAL_ORDER}
conf = RepConfig(hidden_layers=hidden_layers, rep_reader_kwargs=rep_reader_kwargs)

# Load the initial representation method
DEFAULT_METHOD = "Surface"
CLASSIFICATION_NOT_IMPLEMENTED = ["Contrast Vector", "Taylor", "Least Squares"]
CLASSIFICATION_INITIALLY_ENABLED = (DEFAULT_METHOD not in CLASSIFICATION_NOT_IMPLEMENTED)

REP_INFERENCE_KWARGS = dict(
    concept_template_fraction=1.0,
    n_variations_per_concept_level=1,
    remove_instructions=False
) # Takes ~10m for 3 comps, 3 vars, F=0.6, SurfaceRep

print(repr(conf))
conf.representation_method = DEFAULT_METHOD
model, _, rep_manager = load_rep_method(conf, model=base_model, tokenizer=tokenizer)

# Presets
with open("./evaluate/concept_word_presets.json") as preset_file:
    presets = json.load(preset_file)

demo = gr.Blocks()
with demo:
    # UI input components
    gr.Markdown("# Classify and control the amount of a concept within a language model's response", )

    with gr.Row(equal_height=True):
        method = gr.Dropdown(
            choices=["Contrast Vector", "Curve", "Surface", "Taylor", "Least Squares"],
            value=DEFAULT_METHOD,
            label="Method",
            info="Choose a method to use for concept classification and control",
            scale=15
        )
        preset = gr.Dropdown(
            choices=presets.keys(),
            value="None",
            label="Concept preset",
            info="Fill values for predefined concepts",
            scale=15
        )
        concept = gr.Textbox(
            label="Concept",
            info="Adjective that can be used in the form 'you are a highly <concept> person'.",
            scale=20
        )
        concept_antonym = gr.Textbox(
            label="Concept antonym",
            info="Opposite of the concept",
            scale=20
        )
        concept_description = gr.Textbox(
            label="Concept description",
            info="What someone possesing the concept does in the form 'you always <description>'.",
            scale=30
        )
    
    with gr.Row():
        prompt = gr.Textbox(
            label="Prompt",
            info="This text won't be controlled, only the response.",
            scale=30
        )
        generate = gr.Button(value="Generate uncontrolled response", scale=5)
        response = gr.Textbox(value="",
            label="Response",
            info="Type the first few words of the response, which the controlled model "\
                "will follow on from or use the button to get an uncontrolled model response. This response is also used for classification.",
            scale=60
        )
        classify_kwargs = dict(value="Classify uncontrolled response", scale=5)
        classify = gr.Button(visible=CLASSIFICATION_INITIALLY_ENABLED, **classify_kwargs)
    
    gr.Markdown("Concept Control")
    with gr.Row():
        control_strength = gr.Slider(-10, 10, 0,
            label="Control strength",
            info="How much to change the concept by. Negative values decrease the concept.",
            scale=35
        )
        controlled_response = gr.Textbox(
            value="",
            label="Controlled response",
            info="Use the button to get the model's response to the prompt using the selected method",
            scale=60
        )
        control = gr.Button(value="Generate controlled response")
    
    with gr.Column(visible=CLASSIFICATION_INITIALLY_ENABLED) as classification_display:
        gr.Markdown("Concept Classification")
        with gr.Row():
            with gr.Column(scale=60):
                plot_layer = gr.Slider(hidden_layers[1], hidden_layers[0], value=-10, label="Layer", info="Hidden layer to plot", step=1)
                fig = gr.Image()
            with gr.Column(scale=40):
                classification = gr.Label(label="Overall classification score: None", scale=10)
                token_scores = gr.Text(label="Token-level classification", scale=90)
    
    # Run
    generate.click(generate_response, [prompt], [response])
    classify.click(classify_response, inputs=[
        concept, concept_antonym, concept_description, prompt, response, plot_layer, method
    ], outputs=[classification, fig])
    control.click(control_response, inputs=[
        concept, concept_antonym, concept_description, prompt, response, control_strength
    ], outputs=[controlled_response])

    # Event listeners
    # Redraw plot on layer to plot change
    plot_layer.change(plot, [plot_layer, method], [fig])
    # Reload representation method on dropdown change
    method.input(dropdown_load_method, inputs=[method], outputs=[classification_display, classify])

    preset.input(fill_preset_values, inputs=[preset],
        outputs=[concept, concept_antonym, concept_description, prompt, response])
if __name__ == "__main__":
    demo.launch()