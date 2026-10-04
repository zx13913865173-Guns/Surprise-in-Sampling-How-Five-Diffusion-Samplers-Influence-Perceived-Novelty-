# -*- coding: utf-8 -*-
"""
Python cross-check for Paper2 mixed models.
Main analysis is in R (lme4/lmerTest). This script uses statsmodels.
"""

import pandas as pd
import statsmodels.formula.api as smf
from pathlib import Path

Path("outputs/stats").mkdir(parents=True, exist_ok=True)

student = pd.read_csv("data/ratings_student.csv")
expert = pd.read_csv("data/ratings_expert.csv")

# M1: Surprise ~ Sampler + (1|Prompt) + (1|Image) [approximation]
m1 = smf.mixedlm(
    "surprise ~ C(sampler)",
    student,
    groups=student["prompt_id"],
    vc_formula={"Image": "0 + C(image_id)"},
).fit()

with open("outputs/stats/M1_python.txt", "w") as f:
    f.write(m1.summary().as_text())

# M2: add clarity
m2 = smf.mixedlm(
    "surprise ~ C(sampler) + clarity_artifact",
    student,
    groups=student["prompt_id"],
    vc_formula={"Image": "0 + C(image_id)"},
).fit()

with open("outputs/stats/M2_python.txt", "w") as f:
    f.write(m2.summary().as_text())

print("Python cross-check complete.")
