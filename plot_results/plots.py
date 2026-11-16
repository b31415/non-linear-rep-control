"""
Plot the Taylor and Lstsq representations using the coefficients 
calculated for the first controlled layer and first dimension of Lstsq control.

Prompt: You have been waiting in a queue for over two hours and are almost at the front when someone cuts in front of you. You are extremely annoyed. What do you say to them?
    f = 0.6, v = 1, o = 3, layer = -24

Lstsq response for c = 0.2:
    It's important to approach this situation with a calm and respectful demeanor, as responding with anger or confrontation may escalate the situation or create misunderstands. Here are some suggestions for how you might address the situation if someone has inadad an error in the order of the queue and someone has been missed or there has been a misunderstanding regarding the order of the queue and

Taylor response for c = 0.2 (Using same hyperparameters as those used to produce 
the lstsq activations below, but different activations):
    if I have been waiting in a queue for over two hours prior to a situation where someone has been incorrect due a situation involving a situation involving a situation that has been unfortunate for a length of time prior to an issue that has been a matter of a situation that has been a matter for some time prior to a situation that has been a matter for some length prior to a situation that has been a matter

Uncontrolled response:
    I would suggest taking a deep breath before addressing the situation. Here are some possible responses:

    1. Calmly and politely point out the person's mistake: "Excuse me, I believe there's been a mistake. I've been in line before you and have been waiting for quite some time."

    2. Use humor to diffuse the situation:

"""

import seaborn as sns
import math
from scipy.optimize import curve_fit
from numpy.polynomial.polynomial import Polynomial
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def taylor_rep_coefficients(f, order: int = 3):
    """
    Calculates the TaylorRep coefficients for a degree `order`
    polynomial approximation and also returns the set of
    concept levels sampled from the function `f` required
    to calcuate the coefficients.
    """
    coefs = np.zeros((order + 1))
    h = 2 / order
    
    concept_scores_used = []
    for n in range(0, order + 1):
        coef = 0
        for i in range(0, n + 1):
            # Get the term x_0 + (n/2 - i)h where x_0 = 0
            # Round to precision to prevent potential key errors due to floating point errors
            concept_score = round((n/2 - i) * h, 5)
            scalar = ((-1) ** i) * math.comb(n, i)
            # Unexpected concept score    
            if concept_score not in f.keys():
                raise ValueError(f"Concept score {concept_score} not in {concept_levels}")
            # Contribution from prompting a model to act non-neutral
            else:
                coef += scalar * f[concept_score]
            concept_scores_used.append(concept_score)
        coefs[n] = coef / (math.factorial(n) * (h**n))
    return coefs

concept_levels = [1.0, 0.6, 0.2, 0, -0.2, -0.6, -1.0]
taylor_rounded_concept_levels = [1.0, 0.66667, 0.33333, 0.0, -0.33333, -0.66667, -1.0]
# Activation for c = 0 is from concept_demo with Taylor and is not used to fit the lstsq representation for f = 0.6.
activations = [0.0076, 0.0012, 0.0063, -0.0073, -0.0048, -0.0122, -0.0081]
taylor_activations_dict = {c: a for c, a in zip(taylor_rounded_concept_levels, activations)}

# Representations
lstsq_rep = Polynomial([-0.0018,  0.0163,  0.0003, -0.0087])  # Using coefficients calculated  during concept_demo.py
taylor_coefs = taylor_rep_coefficients(taylor_activations_dict, 3)
taylor_rep = Polynomial(taylor_coefs)

# Plot
x_data = np.linspace(-2, 2, 200)
y_data = list(zip(*[(taylor_rep(x), lstsq_rep(x)) for x in x_data]))
labels = ['Taylor', 'Least Squares']
df = pd.DataFrame({
    labels[0]: y_data[0],
    labels[1]: y_data[1],
}, index=x_data)
df.index.name = "Concept Score"

PALETTE = sns.color_palette('tab20', 20)
TAY_COL = PALETTE[0]
LSTSQ_COL = PALETTE[4]
MARKER_COL = PALETTE[8]
# Enable markers for f(x) only to show the points that were sampled to produce
# two of the approximations
def do_common_plot(ax):
    ax.set_ylim([-0.0152, 0.0152])
    plt.gca().yaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    
    ax.set_xlabel("Concept Score", loc="right", labelpad=0)
    ax.set_ylabel("Activation", loc='bottom', labelpad=-45)
    # Move left y-axis and bottom x-axis to centre, passing through (0,0)
    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('center')

    # Eliminate upper and right axes
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    
    # Make axes gray
    ax.spines['left'].set_color('#666')
    ax.spines['bottom'].set_color('#666')

    # Show ticks in the left and lower axes only
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.set_yticks([-0.015, -0.01, -0.005, 0.005, 0.01, 0.015])
    plt.tight_layout()
    return ax

FH = 3.6
FW = 3.3
plt.figure(figsize=(FW, FH))
ax = sns.lineplot(df, palette=[TAY_COL, LSTSQ_COL], dashes=False)
ax = do_common_plot(ax)

# Plot sampled points
ax.set_xticks([-1, -0.5, 0.5, 1])
ax.set_xlim([-1.05, 1.05])
plt.scatter(x=concept_levels, y=activations, color=MARKER_COL, label="Training Set")
plt.legend(frameon=False, loc="lower right", bbox_to_anchor=(1.07, -0.04))
plt.savefig("iclr_overview_plot_training.pdf")

# Inference
x0 = 0.002
xf_taylor = 0
xf_lstsq = 0.005

RED = "#E04F4A"
YELLOW = "#F2B438"
GREEN = "#1B9E77"
plt.figure(figsize=(FW, FH))
ax = sns.lineplot(df["Taylor"], color=TAY_COL, dashes=False)
ax = do_common_plot(ax)

ax.set_xticks([-1, -0.5, 0.5, 1])
plt.scatter(x=[0.0, 0.2], y=[x0, xf_lstsq], c=[YELLOW, GREEN])
ax.set_xlim([-1.05, 1.05])
ax.legend().remove()

plt.savefig("overview_plot_inference.pdf")
plt.show()
pass
