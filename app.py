import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

CNN_PATH = "cnn_model.h5"
CNN_AVAILABLE = os.path.exists(CNN_PATH)
KNN_PATH = "knn_model.pkl"
SVM_PATH = "svm_model.pkl"
# Prefer the local at_risk folder; also support the existing flat GitHub layout.
_RISK_FILES = ("at_risk_model.pkl", "at_risk_scaler.pkl", "at_risk_bin_encoders.pkl",
               "at_risk_features.pkl", "shap_background.pkl")
AT_RISK_DIR = "at_risk" if all(os.path.isfile(os.path.join("at_risk", f)) for f in _RISK_FILES[:4]) else "."
AT_RISK_PATH = os.path.join(AT_RISK_DIR, "at_risk_model.pkl")
AT_RISK_SCALER_PATH = os.path.join(AT_RISK_DIR, "at_risk_scaler.pkl")
AT_RISK_BIN_ENCODERS_PATH = os.path.join(AT_RISK_DIR, "at_risk_bin_encoders.pkl")
AT_RISK_FEATURES_PATH = os.path.join(AT_RISK_DIR, "at_risk_features.pkl")
AT_RISK_AVAILABLE = all(os.path.exists(p) for p in
                        [AT_RISK_PATH, AT_RISK_SCALER_PATH, AT_RISK_BIN_ENCODERS_PATH, AT_RISK_FEATURES_PATH])
SHAP_BG_PATH = os.path.join(AT_RISK_DIR, "shap_background.pkl")
SHAP_AVAILABLE = AT_RISK_AVAILABLE and os.path.exists(SHAP_BG_PATH)

# Parental education: fixed, correct ORDINAL order (NOT alphabetical) -- used only
# by the self-contained At-Risk model, which has its own encoding pipeline.
AT_RISK_EDU_ORDER = {"High School": 0, "Bachelors": 1, "Masters": 2, "PhD": 3}


def sleep_deviation_hours(h: float) -> float:
    """Distance outside the healthy 6-9 hour sleep range; 0 if inside it."""
    if h < 6:
        return 6 - h
    if h > 9:
        return h - 9
    return 0.0

# ------------------------------------------------------------------
# Page setup
# ------------------------------------------------------------------
st.set_page_config(
    page_title="GradeGenius AI",
    page_icon="🎓",
    layout="wide",
)

BANNER_PATH = "banner.png"

# Colors used for each grade badge/chart
GRADE_COLORS = {
    "A": "#2ecc71",   # green
    "B": "#3498db",   # blue
    "C": "#f1c40f",   # yellow
    "D": "#e67e22",   # orange
    "F": "#e74c3c",   # red
}

st.markdown(
    """
    <style>
    /* Compact typography; retain original content geometry. */
    html, body, [class*="css"] { font-size: 18px !important; }
    .stMarkdown p, .stMarkdown li { font-size: 15px !important; line-height: 1.58; }
    .stCaption, div[data-testid="stCaptionContainer"], div[data-testid="stCaptionContainer"] p { font-size: 12px !important; }
    [data-testid="stWidgetLabel"] p, label, .stSelectbox label, .stNumberInput label, .stRadio label {
        font-size: 14px !important; font-weight: 600 !important;
    }
    div[data-baseweb="select"] *, input[type="number"], .stRadio div[role="radiogroup"] label p { font-size: 14px !important; }
    .stButton > button, .stFormSubmitButton > button { font-size: 15px !important; padding: 0.62rem 1.28rem !important; }
    .stButton > button p, .stFormSubmitButton > button p { font-size: 15px !important; }
    div[data-testid="stExpander"] summary, div[data-testid="stExpander"] summary p { font-size: 14px !important; font-weight: 600 !important; }
    [data-testid="stMetricLabel"] p { font-size: 13px !important; }
    [data-testid="stMetricValue"] { font-size: 27px !important; }
    [data-testid="stMainBlockContainer"] h1 { font-size: 31px !important; }
    [data-testid="stMainBlockContainer"] h2 { font-size: 24px !important; }
    [data-testid="stMainBlockContainer"] h3 { font-size: 19px !important; }
    div[data-testid="stMainBlockContainer"] { max-width: 1100px; margin: 0 auto; padding-top: 4.5rem; }

    /* Artwork is painted on the viewport background, never over content. */
    [data-testid="stApp"], .stApp {
        --gg-rail-width: max(0px, calc((100vw - 1100px) / 2 - 28px));
        --gg-tile-height: max(0px, calc((100vh - 144px) / 3));
        --gg-tile-width: min(var(--gg-rail-width), calc(var(--gg-tile-height) * 0.9195));
        --gg-rail-inset: max(12px, calc((100vw - 1100px) / 4 - var(--gg-tile-width) / 2));
        --gg-rendered-height: calc(var(--gg-tile-width) * 1.0875);
        --gg-row-gap: calc((100vh - 96px - 3 * var(--gg-rendered-height)) / 2);
        --gg-row-two: calc(76px + var(--gg-rendered-height) + var(--gg-row-gap));
        --gg-row-three: calc(76px + 2 * var(--gg-rendered-height) + 2 * var(--gg-row-gap));
        background-color: #607e9f !important;
        background-image:
            url("data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAxNjAgMTc0IiB3aWR0aD0iMTYwIiBoZWlnaHQ9IjE3NCI+PGRlZnM+PGxpbmVhckdyYWRpZW50IGlkPSJtZXRhbCIgeDE9IjAiIHkxPSIwIiB4Mj0iMSIgeTI9IjEiPjxzdG9wIHN0b3AtY29sb3I9IiNiNWRkZjgiIC8+PHN0b3Agb2Zmc2V0PSIuMjgiIHN0b3AtY29sb3I9IiM3M2E2ZDYiIC8+PHN0b3Agb2Zmc2V0PSIuNjUiIHN0b3AtY29sb3I9IiM0Yjc5YTkiIC8+PHN0b3Agb2Zmc2V0PSIxIiBzdG9wLWNvbG9yPSIjMjk0YTc1IiAvPjwvbGluZWFyR3JhZGllbnQ+PGxpbmVhckdyYWRpZW50IGlkPSJzY3JlZW4iIHgyPSIxIiB5Mj0iMSI+PHN0b3Agc3RvcC1jb2xvcj0iIzI3NGE3MSIgLz48c3RvcCBvZmZzZXQ9IjEiIHN0b3AtY29sb3I9IiMxMjJlNGUiIC8+PC9saW5lYXJHcmFkaWVudD48cmFkaWFsR3JhZGllbnQgaWQ9Im9yYiIgY3g9Ii4zIiBjeT0iLjI1IiByPSIuOCI+PHN0b3Agc3RvcC1jb2xvcj0iI2YwZmZmZiIgLz48c3RvcCBvZmZzZXQ9Ii40IiBzdG9wLWNvbG9yPSIjODFlMmZmIiAvPjxzdG9wIG9mZnNldD0iMSIgc3RvcC1jb2xvcj0iIzM0NzdiNCIgLz48L3JhZGlhbEdyYWRpZW50PjxsaW5lYXJHcmFkaWVudCBpZD0iZGlzayIgeDI9IjEiIHkyPSIxIj48c3RvcCBzdG9wLWNvbG9yPSIjYjFlOWZjIiAvPjxzdG9wIG9mZnNldD0iMSIgc3RvcC1jb2xvcj0iIzUwODRiNSIgLz48L2xpbmVhckdyYWRpZW50PjxsaW5lYXJHcmFkaWVudCBpZD0idGlsZSIgeDI9IjEiIHkyPSIxIj48c3RvcCBzdG9wLWNvbG9yPSIjNzQ5N2JjIiBzdG9wLW9wYWNpdHk9Ii41MiIgLz48c3RvcCBvZmZzZXQ9IjEiIHN0b3AtY29sb3I9IiMyYzRjNzEiIHN0b3Atb3BhY2l0eT0iLjU2IiAvPjwvbGluZWFyR3JhZGllbnQ+PC9kZWZzPjxnPjxyZWN0IHg9IjUiIHk9IjEyIiB3aWR0aD0iMTUwIiBoZWlnaHQ9IjE1OCIgcng9IjIzIiBmaWxsPSIjMjk0MjVlIiBvcGFjaXR5PSIuNTUiIC8+PHJlY3QgeD0iNSIgeT0iNSIgd2lkdGg9IjE1MCIgaGVpZ2h0PSIxNTgiIHJ4PSIyMyIgZmlsbD0idXJsKCN0aWxlKSIgc3Ryb2tlPSIjYjNkOGY1IiBzdHJva2Utb3BhY2l0eT0iLjciIC8+PHBhdGggZD0iTTI0IDE2aDExMiIgc3Ryb2tlPSIjZDRlYWZmIiBvcGFjaXR5PSIuNTUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgLz48ZyB0cmFuc2Zvcm09InRyYW5zbGF0ZSg3IDEyKSBzY2FsZSguOTEpIj48ZyBzdHJva2U9IiNhOGU4ZmYiIHN0cm9rZS13aWR0aD0iMyIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIj48cGF0aCBkPSJNNDYgMzR2LTE1IE00NiAxMDZ2MTUiIC8+PHBhdGggZD0iTTU4IDM0di0xNSBNNTggMTA2djE1IiAvPjxwYXRoIGQ9Ik03MCAzNHYtMTUgTTcwIDEwNnYxNSIgLz48cGF0aCBkPSJNODIgMzR2LTE1IE04MiAxMDZ2MTUiIC8+PHBhdGggZD0iTTk0IDM0di0xNSBNOTQgMTA2djE1IiAvPjxwYXRoIGQ9Ik0xMDYgMzR2LTE1IE0xMDYgMTA2djE1IiAvPjxwYXRoIGQ9Ik0zNCA0NEgxOSBNMTI2IDQ0aDE1IiAvPjxwYXRoIGQ9Ik0zNCA1NkgxOSBNMTI2IDU2aDE1IiAvPjxwYXRoIGQ9Ik0zNCA2OEgxOSBNMTI2IDY4aDE1IiAvPjxwYXRoIGQ9Ik0zNCA4MEgxOSBNMTI2IDgwaDE1IiAvPjxwYXRoIGQ9Ik0zNCA5MkgxOSBNMTI2IDkyaDE1IiAvPjwvZz48cmVjdCB4PSIzNCIgeT0iMzkiIHdpZHRoPSI5MiIgaGVpZ2h0PSI3NiIgcng9IjEzIiBmaWxsPSIjMjAzYjY1IiAvPjxyZWN0IHg9IjM0IiB5PSIzMCIgd2lkdGg9IjkyIiBoZWlnaHQ9Ijc2IiByeD0iMTMiIGZpbGw9InVybCgjbWV0YWwpIiBzdHJva2U9IiNiM2U1ZmYiIC8+PHJlY3QgeD0iNDkiIHk9IjQ1IiB3aWR0aD0iNjIiIGhlaWdodD0iNDYiIHJ4PSI4IiBmaWxsPSJ1cmwoI3NjcmVlbikiIHN0cm9rZT0iIzc0ZDlmZiIgLz48cGF0aCBkPSJNNTQgNDloNTIiIHN0cm9rZT0iI2Q1ZjZmZiIgb3BhY2l0eT0iLjciIC8+PHRleHQgeD0iODAiIHk9Ijc1IiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmaWxsPSIjZThmY2ZmIiBmb250LXNpemU9IjI1IiBmb250LXdlaWdodD0iODAwIj5BSTwvdGV4dD48L2c+PHRleHQgeD0iODAiIHk9IjE0OCIgZm9udC1zaXplPSIxMiIgZm9udC1mYW1pbHk9IkFyaWFsLHNhbnMtc2VyaWYiIGZvbnQtd2VpZ2h0PSI3MDAiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGZpbGw9IiNmMGY3ZmYiIGxldHRlci1zcGFjaW5nPSIxIj5BSSBQUk9DRVNTT1I8L3RleHQ+PC9nPjwvc3ZnPg=="),
            url("data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAxNjAgMTc0IiB3aWR0aD0iMTYwIiBoZWlnaHQ9IjE3NCI+PGRlZnM+PGxpbmVhckdyYWRpZW50IGlkPSJtZXRhbCIgeDE9IjAiIHkxPSIwIiB4Mj0iMSIgeTI9IjEiPjxzdG9wIHN0b3AtY29sb3I9IiNiNWRkZjgiIC8+PHN0b3Agb2Zmc2V0PSIuMjgiIHN0b3AtY29sb3I9IiM3M2E2ZDYiIC8+PHN0b3Agb2Zmc2V0PSIuNjUiIHN0b3AtY29sb3I9IiM0Yjc5YTkiIC8+PHN0b3Agb2Zmc2V0PSIxIiBzdG9wLWNvbG9yPSIjMjk0YTc1IiAvPjwvbGluZWFyR3JhZGllbnQ+PGxpbmVhckdyYWRpZW50IGlkPSJzY3JlZW4iIHgyPSIxIiB5Mj0iMSI+PHN0b3Agc3RvcC1jb2xvcj0iIzI3NGE3MSIgLz48c3RvcCBvZmZzZXQ9IjEiIHN0b3AtY29sb3I9IiMxMjJlNGUiIC8+PC9saW5lYXJHcmFkaWVudD48cmFkaWFsR3JhZGllbnQgaWQ9Im9yYiIgY3g9Ii4zIiBjeT0iLjI1IiByPSIuOCI+PHN0b3Agc3RvcC1jb2xvcj0iI2YwZmZmZiIgLz48c3RvcCBvZmZzZXQ9Ii40IiBzdG9wLWNvbG9yPSIjODFlMmZmIiAvPjxzdG9wIG9mZnNldD0iMSIgc3RvcC1jb2xvcj0iIzM0NzdiNCIgLz48L3JhZGlhbEdyYWRpZW50PjxsaW5lYXJHcmFkaWVudCBpZD0iZGlzayIgeDI9IjEiIHkyPSIxIj48c3RvcCBzdG9wLWNvbG9yPSIjYjFlOWZjIiAvPjxzdG9wIG9mZnNldD0iMSIgc3RvcC1jb2xvcj0iIzUwODRiNSIgLz48L2xpbmVhckdyYWRpZW50PjxsaW5lYXJHcmFkaWVudCBpZD0idGlsZSIgeDI9IjEiIHkyPSIxIj48c3RvcCBzdG9wLWNvbG9yPSIjNzQ5N2JjIiBzdG9wLW9wYWNpdHk9Ii41MiIgLz48c3RvcCBvZmZzZXQ9IjEiIHN0b3AtY29sb3I9IiMyYzRjNzEiIHN0b3Atb3BhY2l0eT0iLjU2IiAvPjwvbGluZWFyR3JhZGllbnQ+PC9kZWZzPjxnPjxyZWN0IHg9IjUiIHk9IjEyIiB3aWR0aD0iMTUwIiBoZWlnaHQ9IjE1OCIgcng9IjIzIiBmaWxsPSIjMjk0MjVlIiBvcGFjaXR5PSIuNTUiIC8+PHJlY3QgeD0iNSIgeT0iNSIgd2lkdGg9IjE1MCIgaGVpZ2h0PSIxNTgiIHJ4PSIyMyIgZmlsbD0idXJsKCN0aWxlKSIgc3Ryb2tlPSIjYjNkOGY1IiBzdHJva2Utb3BhY2l0eT0iLjciIC8+PHBhdGggZD0iTTI0IDE2aDExMiIgc3Ryb2tlPSIjZDRlYWZmIiBvcGFjaXR5PSIuNTUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgLz48ZyB0cmFuc2Zvcm09InRyYW5zbGF0ZSg3IDEyKSBzY2FsZSguOTEpIj48ZyBzdHJva2U9IiNhMmQ3ZmYiIHN0cm9rZS13aWR0aD0iMS40IiBvcGFjaXR5PSIuNyI+PHBhdGggZD0iTTIzIDQwTDc5IDI1IiAvPjxwYXRoIGQ9Ik0yMyA0MEw3OSA1OCIgLz48cGF0aCBkPSJNMjMgNDBMNzkgOTIiIC8+PHBhdGggZD0iTTIzIDQwTDc5IDEyNSIgLz48cGF0aCBkPSJNMjMgNzZMNzkgMjUiIC8+PHBhdGggZD0iTTIzIDc2TDc5IDU4IiAvPjxwYXRoIGQ9Ik0yMyA3Nkw3OSA5MiIgLz48cGF0aCBkPSJNMjMgNzZMNzkgMTI1IiAvPjxwYXRoIGQ9Ik0yMyAxMTJMNzkgMjUiIC8+PHBhdGggZD0iTTIzIDExMkw3OSA1OCIgLz48cGF0aCBkPSJNMjMgMTEyTDc5IDkyIiAvPjxwYXRoIGQ9Ik0yMyAxMTJMNzkgMTI1IiAvPjxwYXRoIGQ9Ik03OSAyNUwxMzcgNDgiIC8+PHBhdGggZD0iTTc5IDI1TDEzNyAxMDAiIC8+PHBhdGggZD0iTTc5IDU4TDEzNyA0OCIgLz48cGF0aCBkPSJNNzkgNThMMTM3IDEwMCIgLz48cGF0aCBkPSJNNzkgOTJMMTM3IDQ4IiAvPjxwYXRoIGQ9Ik03OSA5MkwxMzcgMTAwIiAvPjxwYXRoIGQ9Ik03OSAxMjVMMTM3IDQ4IiAvPjxwYXRoIGQ9Ik03OSAxMjVMMTM3IDEwMCIgLz48L2c+PGNpcmNsZSBjeD0iMjMiIGN5PSI0MCIgcj0iMTAiIGZpbGw9IiMyNDRjNzAiIC8+PGNpcmNsZSBjeD0iMjMiIGN5PSIzNiIgcj0iMTAiIGZpbGw9InVybCgjb3JiKSIgc3Ryb2tlPSIjYjVmMGZmIiAvPjxjaXJjbGUgY3g9IjIzIiBjeT0iNzYiIHI9IjEwIiBmaWxsPSIjMjQ0YzcwIiAvPjxjaXJjbGUgY3g9IjIzIiBjeT0iNzIiIHI9IjEwIiBmaWxsPSJ1cmwoI29yYikiIHN0cm9rZT0iI2I1ZjBmZiIgLz48Y2lyY2xlIGN4PSIyMyIgY3k9IjExMiIgcj0iMTAiIGZpbGw9IiMyNDRjNzAiIC8+PGNpcmNsZSBjeD0iMjMiIGN5PSIxMDgiIHI9IjEwIiBmaWxsPSJ1cmwoI29yYikiIHN0cm9rZT0iI2I1ZjBmZiIgLz48Y2lyY2xlIGN4PSI3OSIgY3k9IjI1IiByPSIxMCIgZmlsbD0iIzI0NGM3MCIgLz48Y2lyY2xlIGN4PSI3OSIgY3k9IjIxIiByPSIxMCIgZmlsbD0idXJsKCNvcmIpIiBzdHJva2U9IiNiNWYwZmYiIC8+PGNpcmNsZSBjeD0iNzkiIGN5PSI1OCIgcj0iMTAiIGZpbGw9IiMyNDRjNzAiIC8+PGNpcmNsZSBjeD0iNzkiIGN5PSI1NCIgcj0iMTAiIGZpbGw9InVybCgjb3JiKSIgc3Ryb2tlPSIjYjVmMGZmIiAvPjxjaXJjbGUgY3g9Ijc5IiBjeT0iOTIiIHI9IjEwIiBmaWxsPSIjMjQ0YzcwIiAvPjxjaXJjbGUgY3g9Ijc5IiBjeT0iODgiIHI9IjEwIiBmaWxsPSJ1cmwoI29yYikiIHN0cm9rZT0iI2I1ZjBmZiIgLz48Y2lyY2xlIGN4PSI3OSIgY3k9IjEyNSIgcj0iMTAiIGZpbGw9IiMyNDRjNzAiIC8+PGNpcmNsZSBjeD0iNzkiIGN5PSIxMjEiIHI9IjEwIiBmaWxsPSJ1cmwoI29yYikiIHN0cm9rZT0iI2I1ZjBmZiIgLz48Y2lyY2xlIGN4PSIxMzciIGN5PSI0OCIgcj0iMTAiIGZpbGw9IiMyNDRjNzAiIC8+PGNpcmNsZSBjeD0iMTM3IiBjeT0iNDQiIHI9IjEwIiBmaWxsPSJ1cmwoI29yYikiIHN0cm9rZT0iI2I1ZjBmZiIgLz48Y2lyY2xlIGN4PSIxMzciIGN5PSIxMDAiIHI9IjEwIiBmaWxsPSIjMjQ0YzcwIiAvPjxjaXJjbGUgY3g9IjEzNyIgY3k9Ijk2IiByPSIxMCIgZmlsbD0idXJsKCNvcmIpIiBzdHJva2U9IiNiNWYwZmYiIC8+PC9nPjx0ZXh0IHg9IjgwIiB5PSIxNDgiIGZvbnQtc2l6ZT0iMTIiIGZvbnQtZmFtaWx5PSJBcmlhbCxzYW5zLXNlcmlmIiBmb250LXdlaWdodD0iNzAwIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmaWxsPSIjZjBmN2ZmIiBsZXR0ZXItc3BhY2luZz0iMSI+TkVVUkFMIE5FVDwvdGV4dD48L2c+PC9zdmc+"),
            url("data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAxNjAgMTc0IiB3aWR0aD0iMTYwIiBoZWlnaHQ9IjE3NCI+PGRlZnM+PGxpbmVhckdyYWRpZW50IGlkPSJtZXRhbCIgeDE9IjAiIHkxPSIwIiB4Mj0iMSIgeTI9IjEiPjxzdG9wIHN0b3AtY29sb3I9IiNiNWRkZjgiIC8+PHN0b3Agb2Zmc2V0PSIuMjgiIHN0b3AtY29sb3I9IiM3M2E2ZDYiIC8+PHN0b3Agb2Zmc2V0PSIuNjUiIHN0b3AtY29sb3I9IiM0Yjc5YTkiIC8+PHN0b3Agb2Zmc2V0PSIxIiBzdG9wLWNvbG9yPSIjMjk0YTc1IiAvPjwvbGluZWFyR3JhZGllbnQ+PGxpbmVhckdyYWRpZW50IGlkPSJzY3JlZW4iIHgyPSIxIiB5Mj0iMSI+PHN0b3Agc3RvcC1jb2xvcj0iIzI3NGE3MSIgLz48c3RvcCBvZmZzZXQ9IjEiIHN0b3AtY29sb3I9IiMxMjJlNGUiIC8+PC9saW5lYXJHcmFkaWVudD48cmFkaWFsR3JhZGllbnQgaWQ9Im9yYiIgY3g9Ii4zIiBjeT0iLjI1IiByPSIuOCI+PHN0b3Agc3RvcC1jb2xvcj0iI2YwZmZmZiIgLz48c3RvcCBvZmZzZXQ9Ii40IiBzdG9wLWNvbG9yPSIjODFlMmZmIiAvPjxzdG9wIG9mZnNldD0iMSIgc3RvcC1jb2xvcj0iIzM0NzdiNCIgLz48L3JhZGlhbEdyYWRpZW50PjxsaW5lYXJHcmFkaWVudCBpZD0iZGlzayIgeDI9IjEiIHkyPSIxIj48c3RvcCBzdG9wLWNvbG9yPSIjYjFlOWZjIiAvPjxzdG9wIG9mZnNldD0iMSIgc3RvcC1jb2xvcj0iIzUwODRiNSIgLz48L2xpbmVhckdyYWRpZW50PjxsaW5lYXJHcmFkaWVudCBpZD0idGlsZSIgeDI9IjEiIHkyPSIxIj48c3RvcCBzdG9wLWNvbG9yPSIjNzQ5N2JjIiBzdG9wLW9wYWNpdHk9Ii41MiIgLz48c3RvcCBvZmZzZXQ9IjEiIHN0b3AtY29sb3I9IiMyYzRjNzEiIHN0b3Atb3BhY2l0eT0iLjU2IiAvPjwvbGluZWFyR3JhZGllbnQ+PC9kZWZzPjxnPjxyZWN0IHg9IjUiIHk9IjEyIiB3aWR0aD0iMTUwIiBoZWlnaHQ9IjE1OCIgcng9IjIzIiBmaWxsPSIjMjk0MjVlIiBvcGFjaXR5PSIuNTUiIC8+PHJlY3QgeD0iNSIgeT0iNSIgd2lkdGg9IjE1MCIgaGVpZ2h0PSIxNTgiIHJ4PSIyMyIgZmlsbD0idXJsKCN0aWxlKSIgc3Ryb2tlPSIjYjNkOGY1IiBzdHJva2Utb3BhY2l0eT0iLjciIC8+PHBhdGggZD0iTTI0IDE2aDExMiIgc3Ryb2tlPSIjZDRlYWZmIiBvcGFjaXR5PSIuNTUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgLz48ZyB0cmFuc2Zvcm09InRyYW5zbGF0ZSg3IDEyKSBzY2FsZSguOTEpIj48cmVjdCB4PSIxNyIgeT0iMjMiIHdpZHRoPSIxMjYiIGhlaWdodD0iODkiIHJ4PSIxMiIgZmlsbD0iIzIzNDI2YiIgLz48cmVjdCB4PSIxNyIgeT0iMTYiIHdpZHRoPSIxMjYiIGhlaWdodD0iODkiIHJ4PSIxMiIgZmlsbD0idXJsKCNtZXRhbCkiIHN0cm9rZT0iI2IwZTVmZiIgLz48cmVjdCB4PSIyNSIgeT0iMjQiIHdpZHRoPSIxMTAiIGhlaWdodD0iNjUiIHJ4PSI2IiBmaWxsPSJ1cmwoI3NjcmVlbikiIC8+PHBhdGggZD0iTTM2IDQ1aDM1IiBzdHJva2U9IiNhOWVhZmYiIHN0cm9rZS13aWR0aD0iNCIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiAvPjxwYXRoIGQ9Ik0zNiA1NWg2NiIgc3Ryb2tlPSIjNmNlNmM3IiBzdHJva2Utd2lkdGg9IjQiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgLz48cGF0aCBkPSJNMzYgNjVoNDkiIHN0cm9rZT0iI2EzYjVmZiIgc3Ryb2tlLXdpZHRoPSI0IiBzdHJva2UtbGluZWNhcD0icm91bmQiIC8+PHBhdGggZD0iTTM2IDc1aDc1IiBzdHJva2U9IiM2Y2U2YzciIHN0cm9rZS13aWR0aD0iNCIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiAvPjxwYXRoIGQ9Ik02OSAxMDV2MTZoLTE5djdoNjB2LTdIOTF2LTE2IiBmaWxsPSJ1cmwoI21ldGFsKSIgc3Ryb2tlPSIjYjBlNWZmIiAvPjxjaXJjbGUgY3g9IjgwIiBjeT0iOTciIHI9IjIiIGZpbGw9IiM4Y2U5ZmMiIC8+PC9nPjx0ZXh0IHg9IjgwIiB5PSIxNDgiIGZvbnQtc2l6ZT0iMTIiIGZvbnQtZmFtaWx5PSJBcmlhbCxzYW5zLXNlcmlmIiBmb250LXdlaWdodD0iNzAwIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmaWxsPSIjZjBmN2ZmIiBsZXR0ZXItc3BhY2luZz0iMSI+Q09NUFVUSU5HPC90ZXh0PjwvZz48L3N2Zz4="),
            url("data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAxNjAgMTc0IiB3aWR0aD0iMTYwIiBoZWlnaHQ9IjE3NCI+PGRlZnM+PGxpbmVhckdyYWRpZW50IGlkPSJtZXRhbCIgeDE9IjAiIHkxPSIwIiB4Mj0iMSIgeTI9IjEiPjxzdG9wIHN0b3AtY29sb3I9IiNiNWRkZjgiIC8+PHN0b3Agb2Zmc2V0PSIuMjgiIHN0b3AtY29sb3I9IiM3M2E2ZDYiIC8+PHN0b3Agb2Zmc2V0PSIuNjUiIHN0b3AtY29sb3I9IiM0Yjc5YTkiIC8+PHN0b3Agb2Zmc2V0PSIxIiBzdG9wLWNvbG9yPSIjMjk0YTc1IiAvPjwvbGluZWFyR3JhZGllbnQ+PGxpbmVhckdyYWRpZW50IGlkPSJzY3JlZW4iIHgyPSIxIiB5Mj0iMSI+PHN0b3Agc3RvcC1jb2xvcj0iIzI3NGE3MSIgLz48c3RvcCBvZmZzZXQ9IjEiIHN0b3AtY29sb3I9IiMxMjJlNGUiIC8+PC9saW5lYXJHcmFkaWVudD48cmFkaWFsR3JhZGllbnQgaWQ9Im9yYiIgY3g9Ii4zIiBjeT0iLjI1IiByPSIuOCI+PHN0b3Agc3RvcC1jb2xvcj0iI2YwZmZmZiIgLz48c3RvcCBvZmZzZXQ9Ii40IiBzdG9wLWNvbG9yPSIjODFlMmZmIiAvPjxzdG9wIG9mZnNldD0iMSIgc3RvcC1jb2xvcj0iIzM0NzdiNCIgLz48L3JhZGlhbEdyYWRpZW50PjxsaW5lYXJHcmFkaWVudCBpZD0iZGlzayIgeDI9IjEiIHkyPSIxIj48c3RvcCBzdG9wLWNvbG9yPSIjYjFlOWZjIiAvPjxzdG9wIG9mZnNldD0iMSIgc3RvcC1jb2xvcj0iIzUwODRiNSIgLz48L2xpbmVhckdyYWRpZW50PjxsaW5lYXJHcmFkaWVudCBpZD0idGlsZSIgeDI9IjEiIHkyPSIxIj48c3RvcCBzdG9wLWNvbG9yPSIjNzQ5N2JjIiBzdG9wLW9wYWNpdHk9Ii41MiIgLz48c3RvcCBvZmZzZXQ9IjEiIHN0b3AtY29sb3I9IiMyYzRjNzEiIHN0b3Atb3BhY2l0eT0iLjU2IiAvPjwvbGluZWFyR3JhZGllbnQ+PC9kZWZzPjxnPjxyZWN0IHg9IjUiIHk9IjEyIiB3aWR0aD0iMTUwIiBoZWlnaHQ9IjE1OCIgcng9IjIzIiBmaWxsPSIjMjk0MjVlIiBvcGFjaXR5PSIuNTUiIC8+PHJlY3QgeD0iNSIgeT0iNSIgd2lkdGg9IjE1MCIgaGVpZ2h0PSIxNTgiIHJ4PSIyMyIgZmlsbD0idXJsKCN0aWxlKSIgc3Ryb2tlPSIjYjNkOGY1IiBzdHJva2Utb3BhY2l0eT0iLjciIC8+PHBhdGggZD0iTTI0IDE2aDExMiIgc3Ryb2tlPSIjZDRlYWZmIiBvcGFjaXR5PSIuNTUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgLz48ZyB0cmFuc2Zvcm09InRyYW5zbGF0ZSg3IDEyKSBzY2FsZSguOTEpIj48cGF0aCBkPSJNODAgMTB2MTlNMzYgNDVIMTd2MjdoMTlNMTI0IDQ1aDE5djI3aC0xOSIgc3Ryb2tlPSIjYjhlY2ZmIiBzdHJva2Utd2lkdGg9IjQiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgLz48Y2lyY2xlIGN4PSI4MCIgY3k9IjEwIiByPSI2IiBmaWxsPSJ1cmwoI29yYikiIC8+PHJlY3QgeD0iMzQiIHk9IjM2IiB3aWR0aD0iOTIiIGhlaWdodD0iNzMiIHJ4PSIyMiIgZmlsbD0iIzI3NDU2OCIgLz48cmVjdCB4PSIzNCIgeT0iMjkiIHdpZHRoPSI5MiIgaGVpZ2h0PSI3MyIgcng9IjIyIiBmaWxsPSJ1cmwoI21ldGFsKSIgc3Ryb2tlPSIjYmVlYWZmIiAvPjxyZWN0IHg9IjQ2IiB5PSI0NCIgd2lkdGg9IjY4IiBoZWlnaHQ9IjM3IiByeD0iMTMiIGZpbGw9InVybCgjc2NyZWVuKSIgLz48Y2lyY2xlIGN4PSI2NSIgY3k9IjYyIiByPSI3IiBmaWxsPSIjODFmMmU1IiAvPjxjaXJjbGUgY3g9Ijk1IiBjeT0iNjIiIHI9IjciIGZpbGw9IiM4MWYyZTUiIC8+PHBhdGggZD0iTTcwIDkwaDIwIiBzdHJva2U9IiNiY2VmZmYiIHN0cm9rZS13aWR0aD0iMyIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiAvPjxwYXRoIGQ9Ik01MCAxMTZxMzAgMjggNjAgMCIgc3Ryb2tlPSIjOTFlMWZiIiBzdHJva2Utd2lkdGg9IjMiIGZpbGw9Im5vbmUiIC8+PC9nPjx0ZXh0IHg9IjgwIiB5PSIxNDgiIGZvbnQtc2l6ZT0iMTIiIGZvbnQtZmFtaWx5PSJBcmlhbCxzYW5zLXNlcmlmIiBmb250LXdlaWdodD0iNzAwIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmaWxsPSIjZjBmN2ZmIiBsZXR0ZXItc3BhY2luZz0iMSI+QUkgQVNTSVNUQU5UPC90ZXh0PjwvZz48L3N2Zz4="),
            url("data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAxNjAgMTc0IiB3aWR0aD0iMTYwIiBoZWlnaHQ9IjE3NCI+PGRlZnM+PGxpbmVhckdyYWRpZW50IGlkPSJtZXRhbCIgeDE9IjAiIHkxPSIwIiB4Mj0iMSIgeTI9IjEiPjxzdG9wIHN0b3AtY29sb3I9IiNiNWRkZjgiIC8+PHN0b3Agb2Zmc2V0PSIuMjgiIHN0b3AtY29sb3I9IiM3M2E2ZDYiIC8+PHN0b3Agb2Zmc2V0PSIuNjUiIHN0b3AtY29sb3I9IiM0Yjc5YTkiIC8+PHN0b3Agb2Zmc2V0PSIxIiBzdG9wLWNvbG9yPSIjMjk0YTc1IiAvPjwvbGluZWFyR3JhZGllbnQ+PGxpbmVhckdyYWRpZW50IGlkPSJzY3JlZW4iIHgyPSIxIiB5Mj0iMSI+PHN0b3Agc3RvcC1jb2xvcj0iIzI3NGE3MSIgLz48c3RvcCBvZmZzZXQ9IjEiIHN0b3AtY29sb3I9IiMxMjJlNGUiIC8+PC9saW5lYXJHcmFkaWVudD48cmFkaWFsR3JhZGllbnQgaWQ9Im9yYiIgY3g9Ii4zIiBjeT0iLjI1IiByPSIuOCI+PHN0b3Agc3RvcC1jb2xvcj0iI2YwZmZmZiIgLz48c3RvcCBvZmZzZXQ9Ii40IiBzdG9wLWNvbG9yPSIjODFlMmZmIiAvPjxzdG9wIG9mZnNldD0iMSIgc3RvcC1jb2xvcj0iIzM0NzdiNCIgLz48L3JhZGlhbEdyYWRpZW50PjxsaW5lYXJHcmFkaWVudCBpZD0iZGlzayIgeDI9IjEiIHkyPSIxIj48c3RvcCBzdG9wLWNvbG9yPSIjYjFlOWZjIiAvPjxzdG9wIG9mZnNldD0iMSIgc3RvcC1jb2xvcj0iIzUwODRiNSIgLz48L2xpbmVhckdyYWRpZW50PjxsaW5lYXJHcmFkaWVudCBpZD0idGlsZSIgeDI9IjEiIHkyPSIxIj48c3RvcCBzdG9wLWNvbG9yPSIjNzQ5N2JjIiBzdG9wLW9wYWNpdHk9Ii41MiIgLz48c3RvcCBvZmZzZXQ9IjEiIHN0b3AtY29sb3I9IiMyYzRjNzEiIHN0b3Atb3BhY2l0eT0iLjU2IiAvPjwvbGluZWFyR3JhZGllbnQ+PC9kZWZzPjxnPjxyZWN0IHg9IjUiIHk9IjEyIiB3aWR0aD0iMTUwIiBoZWlnaHQ9IjE1OCIgcng9IjIzIiBmaWxsPSIjMjk0MjVlIiBvcGFjaXR5PSIuNTUiIC8+PHJlY3QgeD0iNSIgeT0iNSIgd2lkdGg9IjE1MCIgaGVpZ2h0PSIxNTgiIHJ4PSIyMyIgZmlsbD0idXJsKCN0aWxlKSIgc3Ryb2tlPSIjYjNkOGY1IiBzdHJva2Utb3BhY2l0eT0iLjciIC8+PHBhdGggZD0iTTI0IDE2aDExMiIgc3Ryb2tlPSIjZDRlYWZmIiBvcGFjaXR5PSIuNTUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgLz48ZyB0cmFuc2Zvcm09InRyYW5zbGF0ZSg3IDEyKSBzY2FsZSguOTEpIj48cGF0aCBkPSJNMTMgMTI1aDE0NE0xMyAxMjVWMjAiIHN0cm9rZT0iI2IzZDVmNCIgc3Ryb2tlLXdpZHRoPSIyIiAvPjxwYXRoIGQ9Ik0yNCA5MWwxMCAtNiAxNCA2djI2bC0xNCA2LTEwIC02eiIgZmlsbD0idXJsKCNtZXRhbCkiIHN0cm9rZT0iI2ExZGVmYiIgLz48cGF0aCBkPSJNMzQgODVsMTQgNnYyNmwtMTQgNnoiIGZpbGw9IiMzMTY4OTQiIC8+PHBhdGggZD0iTTU3IDY5bDEwIC02IDE0IDZ2NDhsLTE0IDYtMTAgLTZ6IiBmaWxsPSJ1cmwoI21ldGFsKSIgc3Ryb2tlPSIjYTFkZWZiIiAvPjxwYXRoIGQ9Ik02NyA2M2wxNCA2djQ4bC0xNCA2eiIgZmlsbD0iIzMxNjg5NCIgLz48cGF0aCBkPSJNOTAgNDRsMTAgLTYgMTQgNnY3M2wtMTQgNi0xMCAtNnoiIGZpbGw9InVybCgjbWV0YWwpIiBzdHJva2U9IiNhMWRlZmIiIC8+PHBhdGggZD0iTTEwMCAzOGwxNCA2djczbC0xNCA2eiIgZmlsbD0iIzMxNjg5NCIgLz48cGF0aCBkPSJNMTIzIDI2bDEwIC02IDE0IDZ2OTFsLTE0IDYtMTAgLTZ6IiBmaWxsPSJ1cmwoI21ldGFsKSIgc3Ryb2tlPSIjYTFkZWZiIiAvPjxwYXRoIGQ9Ik0xMzMgMjBsMTQgNnY5MWwtMTQgNnoiIGZpbGw9IiMzMTY4OTQiIC8+PHBhdGggZD0iTTI0IDcwbDM0LTE5IDM1IDQgNDUtMzYiIGZpbGw9Im5vbmUiIHN0cm9rZT0iIzdiZjdkOSIgc3Ryb2tlLXdpZHRoPSI0IiBzdHJva2UtbGluZWNhcD0icm91bmQiIC8+PHBhdGggZD0iTTEyNSAxOWgxM3YxMyIgc3Ryb2tlPSIjN2JmN2Q5IiBzdHJva2Utd2lkdGg9IjQiIGZpbGw9Im5vbmUiIC8+PC9nPjx0ZXh0IHg9IjgwIiB5PSIxNDgiIGZvbnQtc2l6ZT0iMTIiIGZvbnQtZmFtaWx5PSJBcmlhbCxzYW5zLXNlcmlmIiBmb250LXdlaWdodD0iNzAwIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmaWxsPSIjZjBmN2ZmIiBsZXR0ZXItc3BhY2luZz0iMSI+QU5BTFlUSUNTPC90ZXh0PjwvZz48L3N2Zz4="),
            url("data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAxNjAgMTc0IiB3aWR0aD0iMTYwIiBoZWlnaHQ9IjE3NCI+PGRlZnM+PGxpbmVhckdyYWRpZW50IGlkPSJtZXRhbCIgeDE9IjAiIHkxPSIwIiB4Mj0iMSIgeTI9IjEiPjxzdG9wIHN0b3AtY29sb3I9IiNiNWRkZjgiIC8+PHN0b3Agb2Zmc2V0PSIuMjgiIHN0b3AtY29sb3I9IiM3M2E2ZDYiIC8+PHN0b3Agb2Zmc2V0PSIuNjUiIHN0b3AtY29sb3I9IiM0Yjc5YTkiIC8+PHN0b3Agb2Zmc2V0PSIxIiBzdG9wLWNvbG9yPSIjMjk0YTc1IiAvPjwvbGluZWFyR3JhZGllbnQ+PGxpbmVhckdyYWRpZW50IGlkPSJzY3JlZW4iIHgyPSIxIiB5Mj0iMSI+PHN0b3Agc3RvcC1jb2xvcj0iIzI3NGE3MSIgLz48c3RvcCBvZmZzZXQ9IjEiIHN0b3AtY29sb3I9IiMxMjJlNGUiIC8+PC9saW5lYXJHcmFkaWVudD48cmFkaWFsR3JhZGllbnQgaWQ9Im9yYiIgY3g9Ii4zIiBjeT0iLjI1IiByPSIuOCI+PHN0b3Agc3RvcC1jb2xvcj0iI2YwZmZmZiIgLz48c3RvcCBvZmZzZXQ9Ii40IiBzdG9wLWNvbG9yPSIjODFlMmZmIiAvPjxzdG9wIG9mZnNldD0iMSIgc3RvcC1jb2xvcj0iIzM0NzdiNCIgLz48L3JhZGlhbEdyYWRpZW50PjxsaW5lYXJHcmFkaWVudCBpZD0iZGlzayIgeDI9IjEiIHkyPSIxIj48c3RvcCBzdG9wLWNvbG9yPSIjYjFlOWZjIiAvPjxzdG9wIG9mZnNldD0iMSIgc3RvcC1jb2xvcj0iIzUwODRiNSIgLz48L2xpbmVhckdyYWRpZW50PjxsaW5lYXJHcmFkaWVudCBpZD0idGlsZSIgeDI9IjEiIHkyPSIxIj48c3RvcCBzdG9wLWNvbG9yPSIjNzQ5N2JjIiBzdG9wLW9wYWNpdHk9Ii41MiIgLz48c3RvcCBvZmZzZXQ9IjEiIHN0b3AtY29sb3I9IiMyYzRjNzEiIHN0b3Atb3BhY2l0eT0iLjU2IiAvPjwvbGluZWFyR3JhZGllbnQ+PC9kZWZzPjxnPjxyZWN0IHg9IjUiIHk9IjEyIiB3aWR0aD0iMTUwIiBoZWlnaHQ9IjE1OCIgcng9IjIzIiBmaWxsPSIjMjk0MjVlIiBvcGFjaXR5PSIuNTUiIC8+PHJlY3QgeD0iNSIgeT0iNSIgd2lkdGg9IjE1MCIgaGVpZ2h0PSIxNTgiIHJ4PSIyMyIgZmlsbD0idXJsKCN0aWxlKSIgc3Ryb2tlPSIjYjNkOGY1IiBzdHJva2Utb3BhY2l0eT0iLjciIC8+PHBhdGggZD0iTTI0IDE2aDExMiIgc3Ryb2tlPSIjZDRlYWZmIiBvcGFjaXR5PSIuNTUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgLz48ZyB0cmFuc2Zvcm09InRyYW5zbGF0ZSg3IDEyKSBzY2FsZSguOTEpIj48cGF0aCBkPSJNMzAgOTB2MjNjMCAyNCAxMDAgMjQgMTAwIDBWOTAiIGZpbGw9InVybCgjbWV0YWwpIiBzdHJva2U9IiNhOWUyZmYiIC8+PGVsbGlwc2UgY3g9IjgwIiBjeT0iOTAiIHJ4PSI1MCIgcnk9IjE0IiBmaWxsPSJ1cmwoI2Rpc2spIiBzdHJva2U9IiNiOWVmZmYiIC8+PGNpcmNsZSBjeD0iMTE1IiBjeT0iMTA4IiByPSIzIiBmaWxsPSIjNzhmMWNmIiAvPjxwYXRoIGQ9Ik0zMCA1OHYyM2MwIDI0IDEwMCAyNCAxMDAgMFY1OCIgZmlsbD0idXJsKCNtZXRhbCkiIHN0cm9rZT0iI2E5ZTJmZiIgLz48ZWxsaXBzZSBjeD0iODAiIGN5PSI1OCIgcng9IjUwIiByeT0iMTQiIGZpbGw9InVybCgjZGlzaykiIHN0cm9rZT0iI2I5ZWZmZiIgLz48Y2lyY2xlIGN4PSIxMTUiIGN5PSI3NiIgcj0iMyIgZmlsbD0iIzc4ZjFjZiIgLz48cGF0aCBkPSJNMzAgMjZ2MjNjMCAyNCAxMDAgMjQgMTAwIDBWMjYiIGZpbGw9InVybCgjbWV0YWwpIiBzdHJva2U9IiNhOWUyZmYiIC8+PGVsbGlwc2UgY3g9IjgwIiBjeT0iMjYiIHJ4PSI1MCIgcnk9IjE0IiBmaWxsPSJ1cmwoI2Rpc2spIiBzdHJva2U9IiNiOWVmZmYiIC8+PGNpcmNsZSBjeD0iMTE1IiBjeT0iNDQiIHI9IjMiIGZpbGw9IiM3OGYxY2YiIC8+PC9nPjx0ZXh0IHg9IjgwIiB5PSIxNDgiIGZvbnQtc2l6ZT0iMTIiIGZvbnQtZmFtaWx5PSJBcmlhbCxzYW5zLXNlcmlmIiBmb250LXdlaWdodD0iNzAwIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmaWxsPSIjZjBmN2ZmIiBsZXR0ZXItc3BhY2luZz0iMSI+REFUQSBTQ0lFTkNFPC90ZXh0PjwvZz48L3N2Zz4="),
            radial-gradient(ellipse at 12% 12%, #8ba7c6 0%, transparent 48%),
            radial-gradient(ellipse at 88% 82%, #739fa6 0%, transparent 48%),
            linear-gradient(135deg, #607e9f, #536a90 55%, #688a9c) !important;
        background-size: var(--gg-tile-width) auto, var(--gg-tile-width) auto, var(--gg-tile-width) auto,
                         var(--gg-tile-width) auto, var(--gg-tile-width) auto, var(--gg-tile-width) auto,
                         cover, cover, cover !important;
        background-position: left var(--gg-rail-inset) top 76px, left var(--gg-rail-inset) top var(--gg-row-two),
            left var(--gg-rail-inset) top var(--gg-row-three), right var(--gg-rail-inset) top 76px,
            right var(--gg-rail-inset) top var(--gg-row-two), right var(--gg-rail-inset) top var(--gg-row-three),
            center, center, center !important;
        background-repeat: no-repeat !important;
        background-attachment: fixed !important;
    }

    [data-testid="stMain"] { background: transparent !important; }
    [data-testid="stHeader"] { background: linear-gradient(100deg, #c9d9ec, #dcdcf0 48%, #b9dedf) !important; box-shadow: none !important; }
    [data-testid="stHeader"] button { color: #213650 !important; }
    [data-testid="stMainBlockContainer"] {
        background: radial-gradient(ellipse at 0% 8%, #e5d9f2 0%, transparent 50%),
                    radial-gradient(ellipse at 100% 38%, #b9e3df 0%, transparent 55%),
                    linear-gradient(145deg, #c5d9ee 0%, #d4d8ee 48%, #bededb 100%) !important;
        border: 1px solid rgba(190,219,246,.42);
        border-radius: 24px;
        box-shadow: 0 18px 45px rgba(24,44,66,.35), inset 0 1px 0 rgba(255,255,255,.22);
        /* Opaque gradient prevents artwork showing through the page. */
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #456380, #34516e);
        border-right: 1px solid rgba(102,149,255,.15);
    }
    .stAlert { border-radius: 16px; }

    /* Readable ink on the lighter mixed-colour page. */
    [data-testid="stMainBlockContainer"] :is(.stMarkdown, [data-testid="stWidgetLabel"], [data-testid="stMetricLabel"], [data-testid="stMetricValue"], [data-testid="stExpander"] summary, p, li, span, label) {
        color: #213650;
    }

    /* Raised blue-grey form: lighter than the page, never pure white. */
    [data-testid="stForm"] {
        background: linear-gradient(145deg, #dce6f3 0%, #c0d1e8 40%, #aacdd0 100%) !important;
        border: 1px solid #d3e5f5 !important;
        border-bottom: 4px solid #6887a6 !important;
        border-right: 3px solid #7895b2 !important;
        border-radius: 22px !important;
        box-shadow: 10px 14px 28px rgba(20,40,65,.35), -4px -4px 12px rgba(188,215,241,.20), inset 2px 2px 2px rgba(238,250,255,.6), inset -2px -2px 5px rgba(72,103,138,.20) !important;
        padding-top: 0.6rem;
    }
    [data-testid="stMainBlockContainer"] [data-testid="stForm"] :is([data-testid="stWidgetLabel"] p, label, .stMarkdown p, .stMarkdown li, h1, h2, h3, [data-testid="stCaptionContainer"] p) {
        color: #172f4b !important;
    }
    [data-testid="stMainBlockContainer"] [data-testid="stForm"] .gg-section { color: #fff !important; }
    [data-testid="stMainBlockContainer"] [data-testid="stForm"] button p { color: #fff !important; }
    [data-testid="stMainBlockContainer"] [data-testid="stNumberInput"] button {
        color: #eaf4ff !important; background: linear-gradient(145deg, #31527d, #152d4e) !important;
        border: 1px solid #5579a4 !important; box-shadow: inset 0 1px 0 #7292b5, 0 3px 5px #35577766;
    }
    [data-testid="stMainBlockContainer"] [data-baseweb="select"] svg { fill: #d9eaff !important; color: #d9eaff !important; }
    [data-testid="stMainBlockContainer"] [data-baseweb="select"] :is(span, [role="combobox"], input) { color: #f3f8ff !important; }
    [data-baseweb="popover"] [role="listbox"] { background: #233e60 !important; color: #eef6ff !important; }
    [data-baseweb="popover"] [role="option"] { background: #233e60 !important; color: #eef6ff !important; }
    [data-baseweb="popover"] [role="option"]:hover, [data-baseweb="popover"] [aria-selected="true"] { background: #42678e !important; color: #fff !important; }
    [data-testid="stMainBlockContainer"] [data-baseweb="select"]:focus-within > div,
    [data-testid="stMainBlockContainer"] [data-baseweb="input"]:focus-within { outline: 2px solid #91ddff; outline-offset: 2px; }
    [data-testid="stExpander"] {
        background: linear-gradient(145deg, #d7e3f2, #bdd9dc);
        border-radius: 14px !important;
        border-left: 5px solid #3a86ff !important;
        box-shadow: 0 8px 22px rgba(0,0,0,.25);
    }
    [data-testid="stMetric"] {
        background: linear-gradient(145deg, #d7e3f2, #b7cde3);
        border-radius: 16px;
        border: 1px solid rgba(120,168,255,.16);
        box-shadow: 0 10px 26px rgba(0,0,0,.28);
    }

    /* Input widgets: darker 3D neumorphic look, not plain white */
    [data-testid="stMainBlockContainer"] [data-baseweb="select"] > div,
    [data-testid="stMainBlockContainer"] [data-baseweb="input"],
    [data-testid="stMainBlockContainer"] input,
    [data-testid="stMainBlockContainer"] textarea,
    [data-testid="stMainBlockContainer"] div[data-baseweb="select"] > div > div {
        background: linear-gradient(145deg, #182b53, #0d1832) !important;
        color: #f3f8ff !important;
        border: 1px solid rgba(121,168,255,.26) !important;
        border-radius: 15px !important;
        box-shadow:
            inset 3px 3px 8px rgba(255,255,255,.04),
            inset -4px -4px 10px rgba(0,0,0,.42),
            7px 7px 16px rgba(0,0,0,.28),
            -4px -4px 12px rgba(38,67,126,.14) !important;
    }
    [data-testid="stMainBlockContainer"] input::placeholder,
    [data-testid="stMainBlockContainer"] textarea::placeholder { color: #c8d8f7 !important; }
    [data-testid="stMainBlockContainer"] [data-baseweb="select"] svg,
    [data-testid="stMainBlockContainer"] input,
    [data-testid="stMainBlockContainer"] [data-baseweb="input"] input {
        color: #f3f8ff !important;
        caret-color: #f3f8ff !important;
    }
    [data-testid="stMainBlockContainer"] [data-baseweb="select"] > div:hover,
    [data-testid="stMainBlockContainer"] [data-baseweb="input"]:hover,
    [data-testid="stMainBlockContainer"] input:hover,
    [data-testid="stMainBlockContainer"] textarea:hover {
        border-color: rgba(101,196,255,.55) !important;
        box-shadow:
            inset 3px 3px 8px rgba(255,255,255,.05),
            inset -5px -5px 10px rgba(0,0,0,.48),
            0 0 0 1px rgba(93,190,255,.18),
            0 10px 22px rgba(25,126,255,.16) !important;
    }

    /* Floating banner - size untouched */
    div[data-testid="stImage"] img {
        border-radius: 20px;
        box-shadow: 0 12px 30px rgba(80, 60, 200, 0.25);
        animation: floatY 4s ease-in-out infinite;
    }
    @keyframes floatY {
        0%, 100% { transform: translateY(0px); }
        50%      { transform: translateY(-8px); }
    }

    /* Gradient animated title */
    .gg-title {
        font-size: clamp(31px, 4.8vw, 54px);
        line-height: 1.15;
        font-weight: 900;
        text-align: center;
        margin-bottom: 0px;
        background: linear-gradient(90deg, #5835b7, #2059b9, #087b82, #5835b7);
        background-size: 300% auto;
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: titleShine 6s linear infinite;
    }
    @keyframes titleShine {
        0%   { background-position: 0% 50%; }
        100% { background-position: 300% 50%; }
    }
    .gg-subtitle {
        text-align: center;
        font-size: clamp(16px, 2.2vw, 23px);
        font-weight: 700;
        color: #263e65;
        margin-top: 6px;
        text-shadow: 0 1px 2px rgba(255,255,255,.5);
    }
    .gg-tagline {
        text-align: center;
        font-size: clamp(12px, 1.3vw, 14px);
        color: #38546e;
        margin-top: 6px;
        margin-bottom: 10px;
    }

    .gg-section {
        display: inline-block;
        padding: 6px 18px;
        border-radius: 999px;
        background: linear-gradient(90deg, #6f55ff, #3a86ff);
        color: white !important;
        font-weight: 700;
        font-size: 15px;
        margin-bottom: 10px;
        box-shadow: 0 4px 14px rgba(58, 134, 255, 0.32);
    }

    .grade-badge {
        display: inline-block;
        padding: 18px 38px;
        border-radius: 18px;
        font-size: 40px;
        font-weight: 800;
        color: white;
        text-align: center;
        box-shadow: 0 0 25px 4px var(--glow, rgba(52,152,219,0.6)), 0 10px 20px rgba(0,0,0,0.15);
        animation: pulseGlow 2.2s ease-in-out infinite;
        transition: transform 0.25s ease;
        transform-style: preserve-3d;
    }
    .grade-badge:hover {
        transform: perspective(500px) rotateX(8deg) rotateY(-8deg) scale(1.05);
    }
    @keyframes pulseGlow {
        0%, 100% { box-shadow: 0 0 18px 2px var(--glow, rgba(52,152,219,0.55)), 0 10px 20px rgba(0,0,0,0.15); }
        50%      { box-shadow: 0 0 32px 8px var(--glow, rgba(52,152,219,0.85)), 0 10px 20px rgba(0,0,0,0.15); }
    }

    .metric-card {
        background: linear-gradient(145deg, #d7e3f2, #b7cde3);
        border-radius: 16px;
        border: 1px solid rgba(113,162,255,.16);
        padding: 14px;
        box-shadow: 0 10px 24px rgba(0,0,0,.25);
        text-align: center;
        transition: transform 0.25s ease, box-shadow 0.25s ease;
    }
    .metric-card:hover {
        transform: perspective(600px) rotateX(4deg) translateY(-4px);
        box-shadow: 0 14px 28px rgba(106, 17, 203, 0.25);
    }

    .stButton > button, .stFormSubmitButton > button {
        background: linear-gradient(90deg, #7a53ff, #2c7cff, #11cbbf);
        background-size: 200% auto;
        color: white !important;
        font-weight: 700;
        border: none;
        border-radius: 999px;
        box-shadow: 0 6px 18px rgba(37, 117, 252, 0.35);
        transition: all 0.3s ease;
    }
    .stButton > button:hover, .stFormSubmitButton > button:hover {
        background-position: right center;
        transform: translateY(-2px) scale(1.02);
        box-shadow: 0 10px 24px rgba(106, 17, 203, 0.45);
    }

    [data-testid="stMainBlockContainer"] :is(h1,h2,h3,h4,h5,h6) { color: #213650; }
    [data-testid="stMainBlockContainer"] .gg-section,
    [data-testid="stMainBlockContainer"] :is(.stButton,.stFormSubmitButton) button p { color: #fff !important; }
    [data-testid="stMainBlockContainer"] [data-testid="stSlider"] [role="slider"] { background: #31527d; }
    [data-testid="stMainBlockContainer"] [data-testid="stCaptionContainer"] p { color: #38546e; }
    [data-testid="stMainBlockContainer"] [data-testid="stAlert"] p { color: #213650; }

    @media (prefers-reduced-motion: reduce) {
        .gg-title, .grade-badge, div[data-testid="stImage"] img { animation: none !important; }
    }
    @media (max-width: 1350px) {
        [data-testid="stApp"], .stApp { background-image: radial-gradient(ellipse at 15% 12%, #7899bd 0%, transparent 48%), linear-gradient(135deg, #4c6a88, #395875 55%, #536e90) !important; background-size: cover, cover !important; background-position: center !important; }
        [data-testid="stMainBlockContainer"]::before {
            content: ""; display: block; width: 100%; height: auto; aspect-ratio: 960 / 180;
            background: url("data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSI5NjAiIGhlaWdodD0iMTgwIiB2aWV3Qm94PSIwIDAgOTYwIDE4MCIgZm9udC1mYW1pbHk9IkFyaWFsLHNhbnMtc2VyaWYiPjxkZWZzPjxsaW5lYXJHcmFkaWVudCBpZD0ibWV0YWwiIHgxPSIwIiB5MT0iMCIgeDI9IjEiIHkyPSIxIj48c3RvcCBzdG9wLWNvbG9yPSIjYjVkZGY4Ii8+PHN0b3Agb2Zmc2V0PSIuMjgiIHN0b3AtY29sb3I9IiM3M2E2ZDYiLz48c3RvcCBvZmZzZXQ9Ii42NSIgc3RvcC1jb2xvcj0iIzRiNzlhOSIvPjxzdG9wIG9mZnNldD0iMSIgc3RvcC1jb2xvcj0iIzI5NGE3NSIvPjwvbGluZWFyR3JhZGllbnQ+PGxpbmVhckdyYWRpZW50IGlkPSJzY3JlZW4iIHgyPSIxIiB5Mj0iMSI+PHN0b3Agc3RvcC1jb2xvcj0iIzI3NGE3MSIvPjxzdG9wIG9mZnNldD0iMSIgc3RvcC1jb2xvcj0iIzEyMmU0ZSIvPjwvbGluZWFyR3JhZGllbnQ+PHJhZGlhbEdyYWRpZW50IGlkPSJvcmIiIGN4PSIuMyIgY3k9Ii4yNSIgcj0iLjgiPjxzdG9wIHN0b3AtY29sb3I9IiNmMGZmZmYiLz48c3RvcCBvZmZzZXQ9Ii40IiBzdG9wLWNvbG9yPSIjODFlMmZmIi8+PHN0b3Agb2Zmc2V0PSIxIiBzdG9wLWNvbG9yPSIjMzQ3N2I0Ii8+PC9yYWRpYWxHcmFkaWVudD48bGluZWFyR3JhZGllbnQgaWQ9ImRpc2siIHgyPSIxIiB5Mj0iMSI+PHN0b3Agc3RvcC1jb2xvcj0iI2IxZTlmYyIvPjxzdG9wIG9mZnNldD0iMSIgc3RvcC1jb2xvcj0iIzUwODRiNSIvPjwvbGluZWFyR3JhZGllbnQ+PGxpbmVhckdyYWRpZW50IGlkPSJ0aWxlIiB4Mj0iMSIgeTI9IjEiPjxzdG9wIHN0b3AtY29sb3I9IiM3NDk3YmMiIHN0b3Atb3BhY2l0eT0iLjUyIi8+PHN0b3Agb2Zmc2V0PSIxIiBzdG9wLWNvbG9yPSIjMmM0YzcxIiBzdG9wLW9wYWNpdHk9Ii41NiIvPjwvbGluZWFyR3JhZGllbnQ+PC9kZWZzPjxnIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAgMCkiPjxyZWN0IHg9IjUiIHk9IjEyIiB3aWR0aD0iMTUwIiBoZWlnaHQ9IjE1OCIgcng9IjIzIiBmaWxsPSIjMjk0MjVlIiBvcGFjaXR5PSIuNTUiLz48cmVjdCB4PSI1IiB5PSI1IiB3aWR0aD0iMTUwIiBoZWlnaHQ9IjE1OCIgcng9IjIzIiBmaWxsPSJ1cmwoI3RpbGUpIiBzdHJva2U9IiNiM2Q4ZjUiIHN0cm9rZS1vcGFjaXR5PSIuNyIvPjxwYXRoIGQ9Ik0yNCAxNmgxMTIiIHN0cm9rZT0iI2Q0ZWFmZiIgb3BhY2l0eT0iLjU1IiBzdHJva2UtbGluZWNhcD0icm91bmQiLz48ZyB0cmFuc2Zvcm09InRyYW5zbGF0ZSg3IDEyKSBzY2FsZSguOTEpIj48ZyBzdHJva2U9IiNhOGU4ZmYiIHN0cm9rZS13aWR0aD0iMyIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIj48cGF0aCBkPSJNNDYgMzR2LTE1IE00NiAxMDZ2MTUiLz48cGF0aCBkPSJNNTggMzR2LTE1IE01OCAxMDZ2MTUiLz48cGF0aCBkPSJNNzAgMzR2LTE1IE03MCAxMDZ2MTUiLz48cGF0aCBkPSJNODIgMzR2LTE1IE04MiAxMDZ2MTUiLz48cGF0aCBkPSJNOTQgMzR2LTE1IE05NCAxMDZ2MTUiLz48cGF0aCBkPSJNMTA2IDM0di0xNSBNMTA2IDEwNnYxNSIvPjxwYXRoIGQ9Ik0zNCA0NEgxOSBNMTI2IDQ0aDE1Ii8+PHBhdGggZD0iTTM0IDU2SDE5IE0xMjYgNTZoMTUiLz48cGF0aCBkPSJNMzQgNjhIMTkgTTEyNiA2OGgxNSIvPjxwYXRoIGQ9Ik0zNCA4MEgxOSBNMTI2IDgwaDE1Ii8+PHBhdGggZD0iTTM0IDkySDE5IE0xMjYgOTJoMTUiLz48L2c+PHJlY3QgeD0iMzQiIHk9IjM5IiB3aWR0aD0iOTIiIGhlaWdodD0iNzYiIHJ4PSIxMyIgZmlsbD0iIzIwM2I2NSIvPjxyZWN0IHg9IjM0IiB5PSIzMCIgd2lkdGg9IjkyIiBoZWlnaHQ9Ijc2IiByeD0iMTMiIGZpbGw9InVybCgjbWV0YWwpIiBzdHJva2U9IiNiM2U1ZmYiLz48cmVjdCB4PSI0OSIgeT0iNDUiIHdpZHRoPSI2MiIgaGVpZ2h0PSI0NiIgcng9IjgiIGZpbGw9InVybCgjc2NyZWVuKSIgc3Ryb2tlPSIjNzRkOWZmIi8+PHBhdGggZD0iTTU0IDQ5aDUyIiBzdHJva2U9IiNkNWY2ZmYiIG9wYWNpdHk9Ii43Ii8+PHRleHQgeD0iODAiIHk9Ijc1IiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmaWxsPSIjZThmY2ZmIiBmb250LXNpemU9IjI1IiBmb250LXdlaWdodD0iODAwIj5BSTwvdGV4dD48L2c+PHRleHQgeD0iODAiIHk9IjE0OCIgZm9udC1zaXplPSIxMiIgZm9udC1mYW1pbHk9IkFyaWFsLHNhbnMtc2VyaWYiIGZvbnQtd2VpZ2h0PSI3MDAiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGZpbGw9IiNmMGY3ZmYiIGxldHRlci1zcGFjaW5nPSIxIj5BSSBQUk9DRVNTT1I8L3RleHQ+PC9nPjxnIHRyYW5zZm9ybT0idHJhbnNsYXRlKDE2MCAwKSI+PHJlY3QgeD0iNSIgeT0iMTIiIHdpZHRoPSIxNTAiIGhlaWdodD0iMTU4IiByeD0iMjMiIGZpbGw9IiMyOTQyNWUiIG9wYWNpdHk9Ii41NSIvPjxyZWN0IHg9IjUiIHk9IjUiIHdpZHRoPSIxNTAiIGhlaWdodD0iMTU4IiByeD0iMjMiIGZpbGw9InVybCgjdGlsZSkiIHN0cm9rZT0iI2IzZDhmNSIgc3Ryb2tlLW9wYWNpdHk9Ii43Ii8+PHBhdGggZD0iTTI0IDE2aDExMiIgc3Ryb2tlPSIjZDRlYWZmIiBvcGFjaXR5PSIuNTUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPjxnIHRyYW5zZm9ybT0idHJhbnNsYXRlKDcgMTIpIHNjYWxlKC45MSkiPjxnIHN0cm9rZT0iI2EyZDdmZiIgc3Ryb2tlLXdpZHRoPSIxLjQiIG9wYWNpdHk9Ii43Ij48cGF0aCBkPSJNMjMgNDBMNzkgMjUiLz48cGF0aCBkPSJNMjMgNDBMNzkgNTgiLz48cGF0aCBkPSJNMjMgNDBMNzkgOTIiLz48cGF0aCBkPSJNMjMgNDBMNzkgMTI1Ii8+PHBhdGggZD0iTTIzIDc2TDc5IDI1Ii8+PHBhdGggZD0iTTIzIDc2TDc5IDU4Ii8+PHBhdGggZD0iTTIzIDc2TDc5IDkyIi8+PHBhdGggZD0iTTIzIDc2TDc5IDEyNSIvPjxwYXRoIGQ9Ik0yMyAxMTJMNzkgMjUiLz48cGF0aCBkPSJNMjMgMTEyTDc5IDU4Ii8+PHBhdGggZD0iTTIzIDExMkw3OSA5MiIvPjxwYXRoIGQ9Ik0yMyAxMTJMNzkgMTI1Ii8+PHBhdGggZD0iTTc5IDI1TDEzNyA0OCIvPjxwYXRoIGQ9Ik03OSAyNUwxMzcgMTAwIi8+PHBhdGggZD0iTTc5IDU4TDEzNyA0OCIvPjxwYXRoIGQ9Ik03OSA1OEwxMzcgMTAwIi8+PHBhdGggZD0iTTc5IDkyTDEzNyA0OCIvPjxwYXRoIGQ9Ik03OSA5MkwxMzcgMTAwIi8+PHBhdGggZD0iTTc5IDEyNUwxMzcgNDgiLz48cGF0aCBkPSJNNzkgMTI1TDEzNyAxMDAiLz48L2c+PGNpcmNsZSBjeD0iMjMiIGN5PSI0MCIgcj0iMTAiIGZpbGw9IiMyNDRjNzAiLz48Y2lyY2xlIGN4PSIyMyIgY3k9IjM2IiByPSIxMCIgZmlsbD0idXJsKCNvcmIpIiBzdHJva2U9IiNiNWYwZmYiLz48Y2lyY2xlIGN4PSIyMyIgY3k9Ijc2IiByPSIxMCIgZmlsbD0iIzI0NGM3MCIvPjxjaXJjbGUgY3g9IjIzIiBjeT0iNzIiIHI9IjEwIiBmaWxsPSJ1cmwoI29yYikiIHN0cm9rZT0iI2I1ZjBmZiIvPjxjaXJjbGUgY3g9IjIzIiBjeT0iMTEyIiByPSIxMCIgZmlsbD0iIzI0NGM3MCIvPjxjaXJjbGUgY3g9IjIzIiBjeT0iMTA4IiByPSIxMCIgZmlsbD0idXJsKCNvcmIpIiBzdHJva2U9IiNiNWYwZmYiLz48Y2lyY2xlIGN4PSI3OSIgY3k9IjI1IiByPSIxMCIgZmlsbD0iIzI0NGM3MCIvPjxjaXJjbGUgY3g9Ijc5IiBjeT0iMjEiIHI9IjEwIiBmaWxsPSJ1cmwoI29yYikiIHN0cm9rZT0iI2I1ZjBmZiIvPjxjaXJjbGUgY3g9Ijc5IiBjeT0iNTgiIHI9IjEwIiBmaWxsPSIjMjQ0YzcwIi8+PGNpcmNsZSBjeD0iNzkiIGN5PSI1NCIgcj0iMTAiIGZpbGw9InVybCgjb3JiKSIgc3Ryb2tlPSIjYjVmMGZmIi8+PGNpcmNsZSBjeD0iNzkiIGN5PSI5MiIgcj0iMTAiIGZpbGw9IiMyNDRjNzAiLz48Y2lyY2xlIGN4PSI3OSIgY3k9Ijg4IiByPSIxMCIgZmlsbD0idXJsKCNvcmIpIiBzdHJva2U9IiNiNWYwZmYiLz48Y2lyY2xlIGN4PSI3OSIgY3k9IjEyNSIgcj0iMTAiIGZpbGw9IiMyNDRjNzAiLz48Y2lyY2xlIGN4PSI3OSIgY3k9IjEyMSIgcj0iMTAiIGZpbGw9InVybCgjb3JiKSIgc3Ryb2tlPSIjYjVmMGZmIi8+PGNpcmNsZSBjeD0iMTM3IiBjeT0iNDgiIHI9IjEwIiBmaWxsPSIjMjQ0YzcwIi8+PGNpcmNsZSBjeD0iMTM3IiBjeT0iNDQiIHI9IjEwIiBmaWxsPSJ1cmwoI29yYikiIHN0cm9rZT0iI2I1ZjBmZiIvPjxjaXJjbGUgY3g9IjEzNyIgY3k9IjEwMCIgcj0iMTAiIGZpbGw9IiMyNDRjNzAiLz48Y2lyY2xlIGN4PSIxMzciIGN5PSI5NiIgcj0iMTAiIGZpbGw9InVybCgjb3JiKSIgc3Ryb2tlPSIjYjVmMGZmIi8+PC9nPjx0ZXh0IHg9IjgwIiB5PSIxNDgiIGZvbnQtc2l6ZT0iMTIiIGZvbnQtZmFtaWx5PSJBcmlhbCxzYW5zLXNlcmlmIiBmb250LXdlaWdodD0iNzAwIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmaWxsPSIjZjBmN2ZmIiBsZXR0ZXItc3BhY2luZz0iMSI+TkVVUkFMIE5FVDwvdGV4dD48L2c+PGcgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMzIwIDApIj48cmVjdCB4PSI1IiB5PSIxMiIgd2lkdGg9IjE1MCIgaGVpZ2h0PSIxNTgiIHJ4PSIyMyIgZmlsbD0iIzI5NDI1ZSIgb3BhY2l0eT0iLjU1Ii8+PHJlY3QgeD0iNSIgeT0iNSIgd2lkdGg9IjE1MCIgaGVpZ2h0PSIxNTgiIHJ4PSIyMyIgZmlsbD0idXJsKCN0aWxlKSIgc3Ryb2tlPSIjYjNkOGY1IiBzdHJva2Utb3BhY2l0eT0iLjciLz48cGF0aCBkPSJNMjQgMTZoMTEyIiBzdHJva2U9IiNkNGVhZmYiIG9wYWNpdHk9Ii41NSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIi8+PGcgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoNyAxMikgc2NhbGUoLjkxKSI+PHJlY3QgeD0iMTciIHk9IjIzIiB3aWR0aD0iMTI2IiBoZWlnaHQ9Ijg5IiByeD0iMTIiIGZpbGw9IiMyMzQyNmIiLz48cmVjdCB4PSIxNyIgeT0iMTYiIHdpZHRoPSIxMjYiIGhlaWdodD0iODkiIHJ4PSIxMiIgZmlsbD0idXJsKCNtZXRhbCkiIHN0cm9rZT0iI2IwZTVmZiIvPjxyZWN0IHg9IjI1IiB5PSIyNCIgd2lkdGg9IjExMCIgaGVpZ2h0PSI2NSIgcng9IjYiIGZpbGw9InVybCgjc2NyZWVuKSIvPjxwYXRoIGQ9Ik0zNiA0NWgzNSIgc3Ryb2tlPSIjYTllYWZmIiBzdHJva2Utd2lkdGg9IjQiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPjxwYXRoIGQ9Ik0zNiA1NWg2NiIgc3Ryb2tlPSIjNmNlNmM3IiBzdHJva2Utd2lkdGg9IjQiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPjxwYXRoIGQ9Ik0zNiA2NWg0OSIgc3Ryb2tlPSIjYTNiNWZmIiBzdHJva2Utd2lkdGg9IjQiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPjxwYXRoIGQ9Ik0zNiA3NWg3NSIgc3Ryb2tlPSIjNmNlNmM3IiBzdHJva2Utd2lkdGg9IjQiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPjxwYXRoIGQ9Ik02OSAxMDV2MTZoLTE5djdoNjB2LTdIOTF2LTE2IiBmaWxsPSJ1cmwoI21ldGFsKSIgc3Ryb2tlPSIjYjBlNWZmIi8+PGNpcmNsZSBjeD0iODAiIGN5PSI5NyIgcj0iMiIgZmlsbD0iIzhjZTlmYyIvPjwvZz48dGV4dCB4PSI4MCIgeT0iMTQ4IiBmb250LXNpemU9IjEyIiBmb250LWZhbWlseT0iQXJpYWwsc2Fucy1zZXJpZiIgZm9udC13ZWlnaHQ9IjcwMCIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZmlsbD0iI2YwZjdmZiIgbGV0dGVyLXNwYWNpbmc9IjEiPkNPTVBVVElORzwvdGV4dD48L2c+PGcgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoNDgwIDApIj48cmVjdCB4PSI1IiB5PSIxMiIgd2lkdGg9IjE1MCIgaGVpZ2h0PSIxNTgiIHJ4PSIyMyIgZmlsbD0iIzI5NDI1ZSIgb3BhY2l0eT0iLjU1Ii8+PHJlY3QgeD0iNSIgeT0iNSIgd2lkdGg9IjE1MCIgaGVpZ2h0PSIxNTgiIHJ4PSIyMyIgZmlsbD0idXJsKCN0aWxlKSIgc3Ryb2tlPSIjYjNkOGY1IiBzdHJva2Utb3BhY2l0eT0iLjciLz48cGF0aCBkPSJNMjQgMTZoMTEyIiBzdHJva2U9IiNkNGVhZmYiIG9wYWNpdHk9Ii41NSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIi8+PGcgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoNyAxMikgc2NhbGUoLjkxKSI+PHBhdGggZD0iTTgwIDEwdjE5TTM2IDQ1SDE3djI3aDE5TTEyNCA0NWgxOXYyN2gtMTkiIHN0cm9rZT0iI2I4ZWNmZiIgc3Ryb2tlLXdpZHRoPSI0IiBzdHJva2UtbGluZWNhcD0icm91bmQiLz48Y2lyY2xlIGN4PSI4MCIgY3k9IjEwIiByPSI2IiBmaWxsPSJ1cmwoI29yYikiLz48cmVjdCB4PSIzNCIgeT0iMzYiIHdpZHRoPSI5MiIgaGVpZ2h0PSI3MyIgcng9IjIyIiBmaWxsPSIjMjc0NTY4Ii8+PHJlY3QgeD0iMzQiIHk9IjI5IiB3aWR0aD0iOTIiIGhlaWdodD0iNzMiIHJ4PSIyMiIgZmlsbD0idXJsKCNtZXRhbCkiIHN0cm9rZT0iI2JlZWFmZiIvPjxyZWN0IHg9IjQ2IiB5PSI0NCIgd2lkdGg9IjY4IiBoZWlnaHQ9IjM3IiByeD0iMTMiIGZpbGw9InVybCgjc2NyZWVuKSIvPjxjaXJjbGUgY3g9IjY1IiBjeT0iNjIiIHI9IjciIGZpbGw9IiM4MWYyZTUiLz48Y2lyY2xlIGN4PSI5NSIgY3k9IjYyIiByPSI3IiBmaWxsPSIjODFmMmU1Ii8+PHBhdGggZD0iTTcwIDkwaDIwIiBzdHJva2U9IiNiY2VmZmYiIHN0cm9rZS13aWR0aD0iMyIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIi8+PHBhdGggZD0iTTUwIDExNnEzMCAyOCA2MCAwIiBzdHJva2U9IiM5MWUxZmIiIHN0cm9rZS13aWR0aD0iMyIgZmlsbD0ibm9uZSIvPjwvZz48dGV4dCB4PSI4MCIgeT0iMTQ4IiBmb250LXNpemU9IjEyIiBmb250LWZhbWlseT0iQXJpYWwsc2Fucy1zZXJpZiIgZm9udC13ZWlnaHQ9IjcwMCIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZmlsbD0iI2YwZjdmZiIgbGV0dGVyLXNwYWNpbmc9IjEiPkFJIEFTU0lTVEFOVDwvdGV4dD48L2c+PGcgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoNjQwIDApIj48cmVjdCB4PSI1IiB5PSIxMiIgd2lkdGg9IjE1MCIgaGVpZ2h0PSIxNTgiIHJ4PSIyMyIgZmlsbD0iIzI5NDI1ZSIgb3BhY2l0eT0iLjU1Ii8+PHJlY3QgeD0iNSIgeT0iNSIgd2lkdGg9IjE1MCIgaGVpZ2h0PSIxNTgiIHJ4PSIyMyIgZmlsbD0idXJsKCN0aWxlKSIgc3Ryb2tlPSIjYjNkOGY1IiBzdHJva2Utb3BhY2l0eT0iLjciLz48cGF0aCBkPSJNMjQgMTZoMTEyIiBzdHJva2U9IiNkNGVhZmYiIG9wYWNpdHk9Ii41NSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIi8+PGcgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoNyAxMikgc2NhbGUoLjkxKSI+PHBhdGggZD0iTTEzIDEyNWgxNDRNMTMgMTI1VjIwIiBzdHJva2U9IiNiM2Q1ZjQiIHN0cm9rZS13aWR0aD0iMiIvPjxwYXRoIGQ9Ik0yNCA5MWwxMCAtNiAxNCA2djI2bC0xNCA2LTEwIC02eiIgZmlsbD0idXJsKCNtZXRhbCkiIHN0cm9rZT0iI2ExZGVmYiIvPjxwYXRoIGQ9Ik0zNCA4NWwxNCA2djI2bC0xNCA2eiIgZmlsbD0iIzMxNjg5NCIvPjxwYXRoIGQ9Ik01NyA2OWwxMCAtNiAxNCA2djQ4bC0xNCA2LTEwIC02eiIgZmlsbD0idXJsKCNtZXRhbCkiIHN0cm9rZT0iI2ExZGVmYiIvPjxwYXRoIGQ9Ik02NyA2M2wxNCA2djQ4bC0xNCA2eiIgZmlsbD0iIzMxNjg5NCIvPjxwYXRoIGQ9Ik05MCA0NGwxMCAtNiAxNCA2djczbC0xNCA2LTEwIC02eiIgZmlsbD0idXJsKCNtZXRhbCkiIHN0cm9rZT0iI2ExZGVmYiIvPjxwYXRoIGQ9Ik0xMDAgMzhsMTQgNnY3M2wtMTQgNnoiIGZpbGw9IiMzMTY4OTQiLz48cGF0aCBkPSJNMTIzIDI2bDEwIC02IDE0IDZ2OTFsLTE0IDYtMTAgLTZ6IiBmaWxsPSJ1cmwoI21ldGFsKSIgc3Ryb2tlPSIjYTFkZWZiIi8+PHBhdGggZD0iTTEzMyAyMGwxNCA2djkxbC0xNCA2eiIgZmlsbD0iIzMxNjg5NCIvPjxwYXRoIGQ9Ik0yNCA3MGwzNC0xOSAzNSA0IDQ1LTM2IiBmaWxsPSJub25lIiBzdHJva2U9IiM3YmY3ZDkiIHN0cm9rZS13aWR0aD0iNCIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIi8+PHBhdGggZD0iTTEyNSAxOWgxM3YxMyIgc3Ryb2tlPSIjN2JmN2Q5IiBzdHJva2Utd2lkdGg9IjQiIGZpbGw9Im5vbmUiLz48L2c+PHRleHQgeD0iODAiIHk9IjE0OCIgZm9udC1zaXplPSIxMiIgZm9udC1mYW1pbHk9IkFyaWFsLHNhbnMtc2VyaWYiIGZvbnQtd2VpZ2h0PSI3MDAiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGZpbGw9IiNmMGY3ZmYiIGxldHRlci1zcGFjaW5nPSIxIj5BTkFMWVRJQ1M8L3RleHQ+PC9nPjxnIHRyYW5zZm9ybT0idHJhbnNsYXRlKDgwMCAwKSI+PHJlY3QgeD0iNSIgeT0iMTIiIHdpZHRoPSIxNTAiIGhlaWdodD0iMTU4IiByeD0iMjMiIGZpbGw9IiMyOTQyNWUiIG9wYWNpdHk9Ii41NSIvPjxyZWN0IHg9IjUiIHk9IjUiIHdpZHRoPSIxNTAiIGhlaWdodD0iMTU4IiByeD0iMjMiIGZpbGw9InVybCgjdGlsZSkiIHN0cm9rZT0iI2IzZDhmNSIgc3Ryb2tlLW9wYWNpdHk9Ii43Ii8+PHBhdGggZD0iTTI0IDE2aDExMiIgc3Ryb2tlPSIjZDRlYWZmIiBvcGFjaXR5PSIuNTUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPjxnIHRyYW5zZm9ybT0idHJhbnNsYXRlKDcgMTIpIHNjYWxlKC45MSkiPjxwYXRoIGQ9Ik0zMCA5MHYyM2MwIDI0IDEwMCAyNCAxMDAgMFY5MCIgZmlsbD0idXJsKCNtZXRhbCkiIHN0cm9rZT0iI2E5ZTJmZiIvPjxlbGxpcHNlIGN4PSI4MCIgY3k9IjkwIiByeD0iNTAiIHJ5PSIxNCIgZmlsbD0idXJsKCNkaXNrKSIgc3Ryb2tlPSIjYjllZmZmIi8+PGNpcmNsZSBjeD0iMTE1IiBjeT0iMTA4IiByPSIzIiBmaWxsPSIjNzhmMWNmIi8+PHBhdGggZD0iTTMwIDU4djIzYzAgMjQgMTAwIDI0IDEwMCAwVjU4IiBmaWxsPSJ1cmwoI21ldGFsKSIgc3Ryb2tlPSIjYTllMmZmIi8+PGVsbGlwc2UgY3g9IjgwIiBjeT0iNTgiIHJ4PSI1MCIgcnk9IjE0IiBmaWxsPSJ1cmwoI2Rpc2spIiBzdHJva2U9IiNiOWVmZmYiLz48Y2lyY2xlIGN4PSIxMTUiIGN5PSI3NiIgcj0iMyIgZmlsbD0iIzc4ZjFjZiIvPjxwYXRoIGQ9Ik0zMCAyNnYyM2MwIDI0IDEwMCAyNCAxMDAgMFYyNiIgZmlsbD0idXJsKCNtZXRhbCkiIHN0cm9rZT0iI2E5ZTJmZiIvPjxlbGxpcHNlIGN4PSI4MCIgY3k9IjI2IiByeD0iNTAiIHJ5PSIxNCIgZmlsbD0idXJsKCNkaXNrKSIgc3Ryb2tlPSIjYjllZmZmIi8+PGNpcmNsZSBjeD0iMTE1IiBjeT0iNDQiIHI9IjMiIGZpbGw9IiM3OGYxY2YiLz48L2c+PHRleHQgeD0iODAiIHk9IjE0OCIgZm9udC1zaXplPSIxMiIgZm9udC1mYW1pbHk9IkFyaWFsLHNhbnMtc2VyaWYiIGZvbnQtd2VpZ2h0PSI3MDAiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGZpbGw9IiNmMGY3ZmYiIGxldHRlci1zcGFjaW5nPSIxIj5EQVRBIFNDSUVOQ0U8L3RleHQ+PC9nPjwvc3ZnPg==") center / contain no-repeat; margin: 0 0 12px;
            pointer-events: none;
        }
    }
    @media (max-width: 600px) {
        [data-testid="stMainBlockContainer"]::before { height: auto; }
        [data-testid="stForm"] { border-radius: 15px !important; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------
# Load saved model + preprocessing objects (cached, loads only once)
# ------------------------------------------------------------------
@st.cache_resource
def load_files():
    rf_model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    encoders = joblib.load("label_encoders.pkl")
    meta = joblib.load("meta.pkl")
    return rf_model, scaler, encoders, meta


@st.cache_resource
def load_cnn():
    from tensorflow.keras.models import load_model
    return load_model(CNN_PATH, compile=False)


rf_model, scaler, encoders, meta = load_files()


@st.cache_resource
def load_at_risk_bundle():
    """Self-contained pipeline: its own scaler, own encoders, own feature list.
    Deliberately separate from the main grade-prediction pipeline:
      - excludes gender entirely (should never influence risk)
      - encodes parental education by true ordinal level, not alphabetically
      - trained with monotonic constraints so more attendance / more study time /
        better previous grade / higher parental education can never increase risk
    """
    saved_model = joblib.load(AT_RISK_PATH)
    if isinstance(saved_model, dict) and saved_model.get("format") == "xgboost-json-v1":
        from xgboost import XGBClassifier
        risk_model = XGBClassifier()
        risk_model.load_model(bytearray(saved_model["model"]))
    else:
        risk_model = saved_model
    risk_scaler = joblib.load(AT_RISK_SCALER_PATH)
    bin_encoders = joblib.load(AT_RISK_BIN_ENCODERS_PATH)
    risk_features = joblib.load(AT_RISK_FEATURES_PATH)
    return risk_model, risk_scaler, bin_encoders, risk_features


@st.cache_resource
def load_shap_explainer():
    import shap
    background = joblib.load(SHAP_BG_PATH)
    risk_model, _, _, _ = load_at_risk_bundle()
    return shap.KernelExplainer(risk_model.predict_proba, background)

feature_columns = meta["feature_columns"]
categorical = meta["categorical"]
numeric = meta["numeric"]
best_model_name = meta.get("best_model_name", "Random Forest")


def pretty(name: str) -> str:
    return name.replace("_", " ").title()


def hex_to_rgba(hex_color: str, alpha: float = 0.6) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# ------------------------------------------------------------------
# Academic improvement planner (inference only)
# ------------------------------------------------------------------
def grade_input_scaled(values):
    row = pd.DataFrame([values])
    for col in categorical:
        row[col] = encoders[col].transform(row[col])
    return scaler.transform(row[feature_columns])


def grade_result(values, model_name):
    scaled = grade_input_scaled(values)
    if model_name == "CNN":
        probabilities = load_cnn().predict(scaled[..., None], verbose=0)[0]
        classes = np.arange(len(probabilities))
    else:
        model = (rf_model if model_name == "Random Forest" else
                 load_other_grade_model(model_name))
        probabilities = model.predict_proba(scaled)[0]
        classes = model.classes_
    labels = encoders["final_grade"].inverse_transform(classes)
    return str(labels[np.argmax(probabilities)]), dict(zip(labels, probabilities))


@st.cache_resource
def load_other_grade_model(name):
    return joblib.load({"KNN": KNN_PATH, "SVM": SVM_PATH}[name])


def risk_input_scaled(values, bundle):
    _, risk_scaler, binary_encoders, features = bundle
    row = {}
    for col in features:
        if col == "parental_education":
            row[col] = AT_RISK_EDU_ORDER[values[col]]
        elif col == "sleep_deviation":
            row[col] = sleep_deviation_hours(values["sleep_hours"])
        elif col in binary_encoders:
            row[col] = binary_encoders[col].transform([values[col]])[0]
        else:
            row[col] = values[col]
    return risk_scaler.transform(pd.DataFrame([row])[features])


def risk_score(values, bundle):
    return float(bundle[0].predict_proba(risk_input_scaled(values, bundle))[0, 1])


@st.cache_resource
def load_rf_explainer():
    import shap
    return shap.TreeExplainer(rf_model)


@st.cache_data(show_spinner=False)
def rf_explanation(values):
    scaled = grade_input_scaled(values)
    pred = rf_model.predict(scaled)[0]
    idx = list(rf_model.classes_).index(pred)
    grade = encoders["final_grade"].inverse_transform([pred])[0]
    explainer = load_rf_explainer()
    raw = explainer.shap_values(scaled)
    if isinstance(raw, list):
        impacts = np.asarray(raw[idx])[0]
    else:
        arr = np.asarray(raw)
        if arr.shape != (1, len(feature_columns), len(rf_model.classes_)):
            raise ValueError(f"Unexpected Random Forest SHAP shape: {arr.shape}")
        impacts = arr[0, :, idx]
    base = float(np.asarray(explainer.expected_value).reshape(-1)[idx])
    prob = float(rf_model.predict_proba(scaled)[0, idx])
    if not np.isclose(base + impacts.sum(), prob, atol=1e-5):
        raise ValueError("SHAP contributions do not match the model probability.")
    return str(grade), base, prob, impacts


def show_improvement_planner(values, selected, bundle):
    st.markdown('<span class="gg-section">🎯 Personalized Academic Recommendations</span>', unsafe_allow_html=True)
    # Test feasible, actionable one-factor changes. Never alter gender, family
    # background or historical grades to manufacture a recommendation.
    suggestions = []
    baseline_risk = risk_score(values, bundle) if bundle is not None else None
    base_grade, base_probs = grade_result(values, selected)
    def df_risk(probs):
        return float(probs.get("D", 0) + probs.get("F", 0))
    base_df = df_risk(base_probs)
    candidates = [("study_time_hours", values["study_time_hours"] + 1.0, "Try an additional hour of focused study, if feasible."),
                  ("attendance_percent", values["attendance_percent"] + 5.0, "Explore ways to attend more classes and address attendance barriers.")]
    # This is a scenario within the model's training range, not a medical target.
    if values["sleep_hours"] < 6 or values["sleep_hours"] > 9:
        candidates.append(("sleep_hours", 7.5, "Consider a regular, age-appropriate sleep routine; explore this example sleep duration."))
    for col, target, advice in candidates:
        target = float(np.clip(target, numeric[col]["min"], numeric[col]["max"]))
        if np.isclose(values[col], target):
            continue
        alternative = dict(values, **{col: target})
        g, probs = grade_result(alternative, selected)
        score = risk_score(alternative, bundle) if bundle is not None else df_risk(probs)
        baseline = baseline_risk if bundle is not None else base_df
        suggestions.append((baseline - score, col, target, advice, g))
    suggestions.sort(key=lambda item: item[0], reverse=True)
    score_label = "At-Risk score" if bundle is not None else "D/F grade-model score"
    positive = [item for item in suggestions if item[0] > 0.0005]
    if positive:
        for improvement, col, target, advice, g in positive:
            st.markdown(f"**{pretty(col)}: {values[col]:g} → {target:g}** — {advice}")
            st.write(f"Modelled {score_label} reduction: {improvement*100:.2f} percentage points; predicted grade: {g}.")
    else:
        st.info("The tested single-factor changes do not meaningfully reduce the model score. Maintain supportive study habits and discuss individual barriers with an educator.")
    st.caption("Recommendations rank hypothetical model responses, not proven causal effects. They are suggestions for supportive review, not automatic decisions.")

    st.markdown('<span class="gg-section">🔄 What-If Analysis</span>', unsafe_allow_html=True)
    st.write("Explore changes to study time, attendance and sleep while keeping all other student information fixed.")
    version = st.session_state.get("prediction_version", 0)
    scenario = dict(values)
    with st.form(f"what_if_form_{version}"):
        cols = st.columns(3)
        for area, col, step in zip(cols, ["study_time_hours", "attendance_percent", "sleep_hours"], [.1, 1.0, .1]):
            low, high = float(numeric[col]["min"]), float(numeric[col]["max"])
            with area:
                if high > low:
                    scenario[col] = st.slider(pretty(col), low, high, float(values[col]), step=step, key=f"what_if_{version}_{col}")
                else:
                    st.write(f"{pretty(col)}: {low:g} (fixed)")
        apply = st.form_submit_button("Compare scenario", use_container_width=True)
    if apply:
        st.session_state["what_if_result"] = scenario
    alternative = st.session_state.get("what_if_result", values)
    alt_grade, alt_probs = grade_result(alternative, selected)
    c1, c2, c3 = st.columns(3)
    c1.metric("Original grade", base_grade)
    c2.metric("Scenario grade", alt_grade)
    if bundle is not None:
        alternative_risk = risk_score(alternative, bundle)
        c3.metric("Scenario risk score", f"{alternative_risk:.1%}", f"{(alternative_risk-baseline_risk)*100:+.2f} pp", delta_color="inverse")
        st.write(f"At-Risk score: {baseline_risk:.1%} → {alternative_risk:.1%} | Alert: {'At Risk' if alternative_risk >= .30 else 'On Track'} (30% threshold)")
    else:
        c3.metric("Scenario D/F score", f"{df_risk(alt_probs):.1%}", f"{(df_risk(alt_probs)-base_df)*100:+.2f} pp", delta_color="inverse")
    chart = pd.DataFrame({"Original": pd.Series(base_probs)*100, "Scenario": pd.Series(alt_probs)*100}).reindex(sorted(base_probs))
    st.bar_chart(chart)
    st.caption("Hypothetical predictions from the selected trained grade model and separate At-Risk model. Scores are not guaranteed outcomes or calibrated real-world probabilities.")

    st.markdown('<span class="gg-section">🧠 Random Forest Grade Explainability (SHAP)</span>', unsafe_allow_html=True)
    st.write("Explain the Random Forest prediction for the original student, independently of the selected grade model.")
    if st.checkbox("Show Random Forest SHAP explanation", key=f"rf_shap_{version}"):
        try:
            g, base, prob, impacts = rf_explanation(values)
            st.write(f"Random Forest predicted grade: **{g}** | Background score: {base:.1%} | Student score: {prob:.1%}")
            table = pd.DataFrame({"Feature": [pretty(c) for c in feature_columns], "Student value": [str(values[c]) for c in feature_columns], "Impact (pp)": impacts*100})
            table = table.sort_values("Impact (pp)", key=lambda x: x.abs())
            fig, ax = plt.subplots(figsize=(7, 4.3))
            ax.barh(table["Feature"], table["Impact (pp)"], color=["#29966b" if x >= 0 else "#e74c3c" for x in table["Impact (pp)"]])
            ax.axvline(0, color="#777", linewidth=.8)
            ax.set_xlabel(f"Contribution to grade {g} score (percentage points)")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            st.dataframe(table.iloc[::-1].style.format({"Impact (pp)": "{:+.2f}"}), hide_index=True, use_container_width=True)
            st.caption("Positive SHAP values support this predicted grade; negative values oppose it. They are model attributions, not causal effects or measures of student ability.")
        except Exception as exc:
            st.warning(f"Random Forest explanation unavailable: {exc}")


# ------------------------------------------------------------------
# Header
# ------------------------------------------------------------------
if os.path.exists(BANNER_PATH):
    st.image(BANNER_PATH, use_container_width=True)

st.markdown('<div class="gg-title">GradeGenius AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="gg-subtitle">Know Your Performance. Shape Your Future.</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="gg-tagline">Fill in a student\'s details to instantly predict their final grade \u2014 '
    "compares Random Forest, KNN, SVM"
    + (" and CNN." if CNN_AVAILABLE else " (add the CNN model to compare all 4).")
    + "</div>",
    unsafe_allow_html=True,
)
st.divider()

# ------------------------------------------------------------------
# Model choice
# ------------------------------------------------------------------
model_options = ["Random Forest"]
if os.path.exists(KNN_PATH):
    model_options.append("KNN")
if os.path.exists(SVM_PATH):
    model_options.append("SVM")
if CNN_AVAILABLE:
    model_options.append("CNN")

chosen_model = st.radio("Choose a model", model_options, horizontal=True)
if not CNN_AVAILABLE:
    st.caption("CNN model not found — run `train_cnn.py` in Colab and add "
               "`cnn_model.h5` to this folder to unlock it here.")

# ------------------------------------------------------------------
# Input form (built automatically from the training data)
# ------------------------------------------------------------------
with st.form("student_form"):
    st.markdown('<span class="gg-section">📋 Student Information</span>', unsafe_allow_html=True)

    left, right = st.columns(2)
    inputs = {}

    for i, col in enumerate(feature_columns):
        target = left if i % 2 == 0 else right
        with target:
            if col in categorical:
                inputs[col] = st.selectbox(pretty(col), categorical[col])
            else:
                info = numeric[col]
                inputs[col] = st.number_input(
                    pretty(col),
                    min_value=float(info["min"]),
                    max_value=float(info["max"]),
                    value=float(np.clip(round(info["mean"], 1), info["min"], info["max"])),
                )

    submitted = st.form_submit_button("🔮 Predict Final Grade", use_container_width=True)

# ------------------------------------------------------------------
# Prediction (same preprocessing steps as training)
# ------------------------------------------------------------------
if submitted:
    st.session_state["submitted_student"] = dict(inputs)
    st.session_state["submitted_model"] = chosen_model
    st.session_state["prediction_version"] = st.session_state.get("prediction_version", 0) + 1
    st.session_state.pop("what_if_result", None)

if "submitted_student" in st.session_state:
    inputs = st.session_state["submitted_student"]
    chosen_model = st.session_state["submitted_model"]
    row = pd.DataFrame([inputs])

    for col in categorical:
        row[col] = encoders[col].transform(row[col])

    row = row[feature_columns]
    row_scaled = scaler.transform(row)

    if chosen_model == "CNN":
        cnn = load_cnn()
        row_cnn = row_scaled.reshape(row_scaled.shape[0], row_scaled.shape[1], 1)
        proba_all = cnn.predict(row_cnn, verbose=0)[0]
        pred = int(np.argmax(proba_all))
        active_model = None  # CNN doesn't use sklearn's predict_proba path below
    else:
        if chosen_model == "KNN":
            active_model = load_other_grade_model("KNN")
        elif chosen_model == "SVM":
            active_model = load_other_grade_model("SVM")
        else:
            active_model = rf_model
        pred = active_model.predict(row_scaled)[0]
        proba_all = active_model.predict_proba(row_scaled)[0] if hasattr(active_model, "predict_proba") else None

    grade = encoders["final_grade"].inverse_transform([pred])[0]
    color = GRADE_COLORS.get(grade, "#3498db")
    glow = hex_to_rgba(color, 0.65)

    st.divider()
    st.markdown('<span class="gg-section">🏆 Result</span>', unsafe_allow_html=True)

    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown(
            f'<div class="grade-badge" style="background:{color}; --glow:{glow};">{grade}</div>',
            unsafe_allow_html=True,
        )
    with c2:
        if grade in ("D", "F"):
            st.warning("⚠️ This student is at risk. Consider academic support or intervention.")
        elif grade == "C":
            st.info("This student is performing adequately but has room to improve.")
        else:
            st.success("✅ This student is performing well.")

    if proba_all is not None:
        if chosen_model == "CNN":
            class_ids = np.arange(len(proba_all))
        else:
            class_ids = active_model.classes_
        labels = encoders["final_grade"].inverse_transform(class_ids)
        prob_df = pd.DataFrame({"Grade": labels, "Probability (%)": (proba_all * 100).round(2)})
        prob_df = prob_df.set_index("Grade").reindex(sorted(labels))

        st.markdown('<span class="gg-section">📊 Prediction Confidence</span>', unsafe_allow_html=True)
        bar_colors = [GRADE_COLORS.get(g, "#3498db") for g in prob_df.index]

        fig, ax = plt.subplots(figsize=(6, 3.2))
        bars = ax.barh(prob_df.index.astype(str), prob_df["Probability (%)"], color=bar_colors)
        for bar, val in zip(bars, prob_df["Probability (%)"]):
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}%", va="center", fontsize=10, fontweight="bold")
        ax.set_xlim(0, 110)
        ax.set_xlabel("Probability (%)")
        ax.invert_yaxis()
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.dataframe(prob_df, use_container_width=True)

    # ------------------------------------------------------------------
    # At-Risk Alert — separate binary model (D/F vs A/B/C), tuned for
    # catching at-risk students early rather than naming the exact grade
    # ------------------------------------------------------------------
    risk_bundle = None
    if AT_RISK_AVAILABLE:
        try:
            risk_bundle = load_at_risk_bundle()
        except Exception as exc:
            st.warning("At-Risk model could not be loaded. Run train_at_risk.py in the same Python environment and restart the app.")
            st.caption(f"Model loading error: {exc}")
    else:
        st.warning("At-Risk model files are missing. Check the at_risk folder.")
    if risk_bundle is not None:
        risk_model, risk_scaler, risk_bin_encoders, risk_features = risk_bundle

        # Build this model's own feature row from the same form inputs:
        # gender excluded, parental education re-mapped to true ordinal order.
        risk_row = {}
        for col in risk_features:
            if col == "parental_education":
                risk_row[col] = AT_RISK_EDU_ORDER[inputs["parental_education"]]
            elif col == "sleep_deviation":
                risk_row[col] = sleep_deviation_hours(inputs["sleep_hours"])
            elif col in risk_bin_encoders:
                risk_row[col] = risk_bin_encoders[col].transform([inputs[col]])[0]
            else:
                risk_row[col] = inputs[col]
        risk_df = pd.DataFrame([risk_row])[risk_features]
        risk_row_scaled = risk_scaler.transform(risk_df)

        risk_proba = risk_model.predict_proba(risk_row_scaled)[0][1]  # P(at risk)
        # Threshold lowered from 0.5 to 0.3: for an early-warning system, missing a
        # genuinely at-risk student is worse than a false alarm, so recall is
        # prioritized over precision here. The existing 0.30 threshold is retained.
        is_at_risk = risk_proba >= 0.3

        st.markdown('<span class="gg-section">🚨 At-Risk Alert</span>', unsafe_allow_html=True)
        if is_at_risk:
            st.error(
                f"⚠️ **At Risk** \u2014 model risk score: **{risk_proba*100:.0f}%**. "
                "Early support (tutoring, attendance "
                "follow-up, study-habit coaching) is recommended now, before the final exam."
            )
        else:
            st.success(
                f"✅ **On Track** \u2014 model risk score: "
                f"**{risk_proba*100:.0f}%** based on current habits and history."
            )
        st.caption("Risk threshold: 30%. Scores are not calibrated real-world probabilities; use them to guide supportive review, not automatic decisions.")
        if SHAP_AVAILABLE:
            with st.expander("🔍 Why this result? Click to see which factors mattered"):
                st.markdown(
                    "This alert comes from a dedicated early-warning model trained specifically "
                    "to catch at-risk students (merging D and F grades together) using the expanded "
                    "25,000-student dataset. **Gender is deliberately excluded** from this model, and Attendance, "
                    "Study Time, Previous Grade and Parental Education are mathematically "
                    "constrained so that a *better* value can never increase the risk estimate. "
                    "Below is the breakdown of **this specific student's** result: which of the "
                    f"{len(risk_features)} factors pushed the prediction toward *At Risk*, and "
                    "which pushed it toward *On Track*."
                )

                explainer = load_shap_explainer()
                shap_vals = explainer.shap_values(risk_row_scaled, nsamples=100, silent=True)
                # Older SHAP returns a list per class; newer SHAP returns an array.
                if isinstance(shap_vals, list):
                    contributions = np.asarray(shap_vals[1])[0]
                else:
                    contributions = np.asarray(shap_vals)[0, :, 1]

                feat_labels = ["Sleep Hours" if c == "sleep_deviation" else pretty(c) for c in risk_features]
                student_values = [str(inputs["sleep_hours"]) if c == "sleep_deviation"
                                   else str(inputs[c]) for c in risk_features]
                exp_df = pd.DataFrame({
                    "Feature": feat_labels,
                    "Student's Value": student_values,
                    "Impact": contributions,
                }).sort_values("Impact", key=lambda s: s.abs(), ascending=True)

                bar_colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in exp_df["Impact"]]
                fig2, ax2 = plt.subplots(figsize=(6.5, 4.2))
                ax2.barh(exp_df["Feature"], exp_df["Impact"], color=bar_colors)
                ax2.axvline(0, color="#999", linewidth=0.8)
                ax2.set_xlabel("\u2190 pushes toward On Track   |   pushes toward At Risk \u2192")
                fig2.tight_layout()
                st.pyplot(fig2)
                plt.close(fig2)

                display_df = exp_df.sort_values("Impact", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)
                display_df["Direction"] = np.where(display_df["Impact"] > 0, "\u26a0\ufe0f Toward At Risk", "\u2705 Toward On Track")
                st.dataframe(
                    display_df[["Feature", "Student's Value", "Direction"]],
                    use_container_width=True, hide_index=True,
                )

                top = display_df.iloc[0]
                top_feature = top["Feature"]
                top_value = top["Student's Value"]
                st.markdown(
                    f"**Biggest factor:** *{top_feature}* (value: {top_value}) "
                    f"was the strongest contributor to this result."
                )

    st.divider()
    try:
        show_improvement_planner(inputs, chosen_model, risk_bundle)
    except Exception as exc:
        st.warning(f"Academic improvement planner unavailable: {exc}")

st.divider()

model_list_label = "Random Forest vs KNN vs SVM vs CNN" if CNN_AVAILABLE else "Random Forest vs KNN vs SVM"
with st.expander(f"📈 View model comparison ({model_list_label})"):
    if os.path.exists("Model_Comparison.csv"):
        comp = pd.read_csv("Model_Comparison.csv")
        st.dataframe(comp.style.highlight_max(axis=0, subset=["Accuracy", "Precision", "Recall", "F1 Score"],
                                               color="#d4f7dc").format({m: "{:.2%}" for m in ["Accuracy", "Precision", "Recall", "F1 Score"]}), use_container_width=True)
        if os.path.exists("Model_Comparison_Grouped.png"):
            st.image("Model_Comparison_Grouped.png", use_container_width=True)
    else:
        st.write("Comparison file not found — add Model_Comparison.csv to this folder.")

with st.expander("🔍 View feature importance"):
    if os.path.exists("Feature_Importance.png"):
        st.image("Feature_Importance.png", use_container_width=True)
    else:
        st.write("Feature importance chart not found — add Feature_Importance.png to this folder.")

if CNN_AVAILABLE:
    with st.expander("🧠 View CNN training curves & confusion matrix"):
        for img in ["CNN_Training_Curves.png", "CNN_Confusion_Matrix.png"]:
            if os.path.exists(img):
                st.image(img, use_container_width=True)

st.caption("M.Tech Dissertation Project | Abhishek Chakraborty | Brainware University")
