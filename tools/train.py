import argparse
import glob
import os
import numpy as np

N_DIR = 13
STEP_DEG = 15
MAX_RANGE = 250
OPEN_DIST = 45
STOP_DIST = 25
SCORE_CAP = 150
HALF_WIDTH = 12.25
DEV_PENALTY = 0.6
CLASS_BOXED = N_DIR
N_CLASS = N_DIR + 1
IN_SCALE = float(MAX_RANGE)

ANG = np.deg2rad(np.arange(N_DIR) * STEP_DEG)
REL_DEG = np.arange(N_DIR) * STEP_DEG - 90

_dA = (np.arange(N_DIR)[None, :] - np.arange(N_DIR)[:, None]) * np.deg2rad(STEP_DEG)
_COS, _SIN = np.cos(_dA), np.abs(np.sin(_dA))

def clearance(d):
    d = d.astype(np.float64)[:, None, :]
    along = d * _COS[None]
    lateral = d * _SIN[None]
    blocks = (d < MAX_RANGE) & (along > 1.25) & (lateral < HALF_WIDTH)
    along = np.where(blocks, along, MAX_RANGE)
    return np.floor(along.min(axis = 2))

def rule_class(d):
    clr = clearance(d)
    score = np.minimum(clr, SCORE_CAP) - DEV_PENALTY * np.abs(REL_DEG)[None]
    score = np.where(clr >= OPEN_DIST, score, -np.inf)
    cls = score.argmax(axis = 1)
    cls[np.isneginf(score.max(axis = 1))] = CLASS_BOXED
    return cls


  
