# =========================================================
# Streamlit 실시간 불량 탐지 대시보드 - 경량 버전
# - 불필요한 빈 박스 제거
# - 깜빡임 효과 제거
# - 선택한 그래프만 표시
# - OK1, OK2, NG1, NG2 데이터를 하나의 스트림처럼 연결
# =========================================================

import html
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM


# =========================================================
# 0. 기본 설정
# =========================================================
st.set_page_config(
    page_title="Real-time Welding Monitor Light",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_CSS = """
<style>
html, body, [class*="css"] {
    font-family: Arial, Helvetica, sans-serif;
}

.stApp {
    background: #f7f8fa;
}

.block-container {
    padding-top: 2.2rem;
    padding-bottom: 2rem;
    max-width: 1450px;
}

.title {
    font-size: 30px;
    font-weight: 800;
    color: #111827;
    line-height: 1.35;
    margin-top: 0;
    margin-bottom: 14px;
    padding-top: 4px;
    overflow: visible;
}

.subtitle {
    font-size: 14px;
    color: #667085;
    margin-bottom: 18px;
}

.state-banner-normal {
    background: #ecfdf3;
    color: #027a48;
    border: 2px solid #abefc6;
    border-radius: 12px;
    padding: 12px 16px;
    font-size: 18px;
    font-weight: 700;
    transition: background-color 120ms ease, border-color 120ms ease;
}

.state-banner-alert {
    background: #fef3f2;
    color: #b42318;
    border: 2px solid #f04438;
    border-radius: 12px;
    padding: 12px 16px;
    font-size: 18px;
    font-weight: 800;
    transition: background-color 120ms ease, border-color 120ms ease;
}

.status-wrap {
    background: #ffffff;
    border: 1px solid #d0d5dd;
    border-radius: 16px;
    padding: 16px;
    margin-top: 10px;
    margin-bottom: 12px;
    box-shadow: 0 1px 3px rgba(16,24,40,0.04);
}

.status-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
}

.status-card {
    background: #ffffff;
    border: 2px solid #e4e7ec;
    border-radius: 14px;
    padding: 18px 22px;
    min-height: 96px;
    display: flex;
    align-items: center;
    transition: background-color 120ms ease, border-color 120ms ease;
}

.status-card.alert {
    border-color: #fda29b;
    background: #fff7f7;
}

.dot {
    width: 28px;
    height: 28px;
    border-radius: 50%;
    display: inline-block;
    margin-right: 18px;
    flex-shrink: 0;
}

.dot-blue { background: #4f5bd5; }
.dot-red { background: #ff3b30; }

.status-text {
    font-size: 26px;
    font-weight: 800;
    color: #111827;
}

.note {
    text-align: center;
    color: #667085;
    font-size: 12px;
    margin-top: 10px;
}

.info-card {
    background: #ffffff;
    border: 1px solid #d0d5dd;
    border-radius: 14px;
    padding: 12px 14px;
    box-shadow: 0 1px 3px rgba(16,24,40,0.04);
}

.chart-card {
    background: #ffffff;
    border: 1px solid #d0d5dd;
    border-radius: 16px;
    padding: 4px 10px 2px 10px;
    margin-top: 0;
    margin-bottom: 14px;
    box-shadow: 0 1px 3px rgba(16,24,40,0.04);
    transition: background-color 120ms ease, border-color 120ms ease;
}

.chart-card.alert {
    border-color: #fda29b;
    background: #fffafa;
}



div.stButton > button {
    min-height: 42px;
    font-size: 15px;
    font-weight: 600;
    line-height: 1.2;
}

</style>
"""

st.markdown(BASE_CSS, unsafe_allow_html=True)


def make_flash_overlay_html(flash_id: int) -> str:
    """
    불량 발생 시 화면 전체를 짧고 부드럽게 강조하는 안정화 버전.

    핵심:
    - Streamlit fragment 내부에 overlay를 넣지 않는다.
    - components.html의 JavaScript를 이용해 브라우저 parent document body에 overlay를 직접 붙인다.
    - overlay는 animationend 또는 안전 타이머로 자동 제거된다.
    - 연속 불량에서는 Python 쪽 쿨다운으로 overlay가 과도하게 중첩되지 않는다.
    """
    return f"""
    <script>
    (function() {{
        const doc = window.parent.document;
        const overlayId = 'defect-flash-overlay-{flash_id}';

        if (doc.getElementById(overlayId)) {{
            return;
        }}

        const styleId = 'defect-flash-style';
        let style = doc.getElementById(styleId);

        if (!style) {{
            style = doc.createElement('style');
            style.id = styleId;
            style.innerHTML = `
                @keyframes defectFlashStable {{
                    0%   {{ opacity: 0; }}
                    10%  {{ opacity: 0.34; }}
                    28%  {{ opacity: 0; }}
                    46%  {{ opacity: 0.26; }}
                    64%  {{ opacity: 0; }}
                    100% {{ opacity: 0; }}
                }}

                .defect-flash-overlay {{
                    position: fixed;
                    inset: 0;
                    background: rgba(255, 59, 48, 0.30);
                    z-index: 2147483647;
                    pointer-events: none;
                    opacity: 0;
                    animation: defectFlashStable 1.15s ease-out 1;
                    will-change: opacity;
                    contain: paint;
                }}
            `;
            doc.head.appendChild(style);
        }}

        const overlay = doc.createElement('div');
        overlay.id = overlayId;
        overlay.className = 'defect-flash-overlay';
        overlay.setAttribute('data-flash-id', '{flash_id}');

        doc.body.appendChild(overlay);

        overlay.addEventListener('animationend', function() {{
            overlay.remove();
        }}, {{ once: true }});

        window.setTimeout(function() {{
            if (overlay && overlay.isConnected) {{
                overlay.remove();
            }}
        }}, 1800);
    }})();
    </script>
    """


def make_scroll_position_guard_html(render_id: int, enabled: bool) -> str:
    enabled_text = "true" if enabled else "false"
    return f"""
    <script>
    (function() {{
        const win = window.parent;
        const doc = win.document;
        const key = 'welding-monitor-scroll-y';
        const enabled = {enabled_text};

        function getY() {{
            return win.scrollY || doc.documentElement.scrollTop || doc.body.scrollTop || 0;
        }}

        function setY(y) {{
            win.scrollTo({{ top: y, left: 0, behavior: 'auto' }});
        }}

        if (!win.__weldingScrollGuardInstalled) {{
            win.__weldingScrollGuardInstalled = true;
            win.addEventListener('scroll', function() {{
                const y = getY();
                if (y > 10) {{
                    win.sessionStorage.setItem(key, String(y));
                }}
            }}, {{ passive: true }});
        }}

        const currentY = getY();
        if (currentY > 10) {{
            win.sessionStorage.setItem(key, String(currentY));
        }}

        if (!enabled) {{
            return;
        }}

        const savedY = Number(win.sessionStorage.getItem(key) || 0);
        if (savedY > 10 && getY() < 10) {{
            [0, 40, 120, 260, 520].forEach(function(delay) {{
                win.setTimeout(function() {{
                    if (getY() < 10) {{
                        setY(savedY);
                    }}
                }}, delay);
            }});
        }}
    }})();
    </script>
    """


# =========================================================
# 1. 유틸
# =========================================================
def read_csv_clean(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


def find_first_file(patterns, search_dirs):
    for folder in search_dirs:
        folder = Path(folder)
        for pattern in patterns:
            hits = sorted(folder.glob(pattern))
            if hits:
                return hits[0]
    return None


@st.cache_data(show_spinner=False)
def load_all_data():
    app_dir = Path(__file__).parent if "__file__" in globals() else Path.cwd()
    search_dirs = [app_dir, Path.cwd(), Path("/mnt/data")]
    raw_test_dir = (
        Path.home()
        / "Downloads"
        / "Dataset_전자부품(배터리팩)_예지보전_AI_데이터셋"
        / "data"
        / "raw_data"
        / "test"
    )
    preprocessed_test_dir = (
        Path.home()
        / "Downloads"
        / "Dataset_전자부품(배터리팩)_예지보전_AI_데이터셋"
        / "data"
        / "preprocessed"
        / "test"
    )

    file_patterns = {
        "train": ["df_train_wt*.csv", "df_train_bfsc*.csv", "Training_Data*.csv"],
        "ok1": ["df_ok_1_wt*.csv", "df_ok_1_bfsc*.csv", "WeldingTest_01_OK*.csv"],
        "ok2": ["df_ok_2_wt*.csv", "df_ok_2_bfsc*.csv", "WeldingTest_02_OK*.csv"],
        "ng1": ["df_ng_1_wt*.csv", "df_ng_1_bfsc*.csv", "WeldingTest_03_NG*.csv"],
        "ng2": ["df_ng_2_wt*.csv", "df_ng_2_bfsc*.csv", "WeldingTest_04_NG*.csv"],
        "label_ng1": ["WeldingTest_03_NG_Label*.csv"],
        "label_ng2": ["WeldingTest_04_NG_Label*.csv"],
    }

    paths = {k: find_first_file(v, search_dirs) for k, v in file_patterns.items()}
    test_data_overrides = {
        "train": app_dir / "Training_Data_clean.csv",
        "ok1": raw_test_dir / "WeldingTest_01_OK.csv",
        "ok2": raw_test_dir / "WeldingTest_02_OK.csv",
        "ng1": raw_test_dir / "WeldingTest_03_NG.csv",
        "ng2": raw_test_dir / "WeldingTest_04_NG.csv",
        "label_ng1": preprocessed_test_dir / "WeldingTest_03_NG_Label.csv",
        "label_ng2": preprocessed_test_dir / "WeldingTest_04_NG_Label.csv",
    }

    for key, override_path in test_data_overrides.items():
        if override_path.exists():
            paths[key] = override_path

    required = ["train", "ok1", "ok2", "ng1", "ng2"]
    missing = [key for key in required if paths[key] is None]

    if missing:
        rng = np.random.default_rng(42)
        n_train = 2000
        train = pd.DataFrame({
            "PageNo": np.arange(1, n_train + 1),
            "Speed": rng.normal(250, 5, n_train).round(2),
            "Length": rng.normal(241, 0.8, n_train).round(2),
            "RealPower": rng.normal(1700, 120, n_train).round(2),
            "SetPower": rng.choice([35, 82, 83, 84, 85], n_train),
            "GateOnTime": rng.normal(1150, 180, n_train).round(2),
            "WorkingTime": pd.date_range("2024-01-01 09:00:00", periods=n_train, freq="3s"),
            "cycle_id": np.repeat(np.arange(1, n_train // 40 + 2), 40)[:n_train],
            "order": np.arange(1, n_train + 1),
        })

        def make_test(name, n, defect_ratio, shift):
            df = train.sample(n, replace=True, random_state=len(name)).copy().reset_index(drop=True)
            df["WorkingTime"] = pd.date_range("2024-01-02 09:00:00", periods=n, freq="3s")
            df["order"] = np.arange(1, n + 1)
            df["dataset"] = name
            y = (rng.random(n) < defect_ratio).astype(int)
            df.loc[y == 1, "RealPower"] = df.loc[y == 1, "RealPower"] + shift
            return df, y

        ok1, _ = make_test("OK1", 300, 0.00, 0)
        ok2, _ = make_test("OK2", 300, 0.00, 0)
        ng1, y_ng1 = make_test("NG1", 300, 0.08, 750)
        ng2, y_ng2 = make_test("NG2", 300, 0.65, 950)
        return train, ok1, ok2, ng1, ng2, y_ng1, y_ng2, paths, missing

    train = read_csv_clean(paths["train"])
    ok1 = read_csv_clean(paths["ok1"])
    ok2 = read_csv_clean(paths["ok2"])
    ng1 = read_csv_clean(paths["ng1"])
    ng2 = read_csv_clean(paths["ng2"])

    y_ng1 = read_csv_clean(paths["label_ng1"])["label"].astype(int).values if paths["label_ng1"] else np.zeros(len(ng1), dtype=int)
    y_ng2 = read_csv_clean(paths["label_ng2"])["label"].astype(int).values if paths["label_ng2"] else np.zeros(len(ng2), dtype=int)

    return train, ok1, ok2, ng1, ng2, y_ng1, y_ng2, paths, []


def prepare_time_and_order(df: pd.DataFrame, dataset_name: str, offset_seconds: int = 0):
    df = df.copy()
    df.columns = df.columns.str.strip()

    for col in ["PageNo", "Speed", "Length", "RealPower", "SetPower", "SetFrequency", "SetDuty", "GateOnTime", "cycle_id", "order"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "order" not in df.columns:
        df["order"] = np.arange(1, len(df) + 1)

    if "cycle_id" not in df.columns:
        df["cycle_id"] = (df["order"] - 1) // 40 + 1

    if "WorkingTime" in df.columns:
        df["WorkingTime"] = pd.to_datetime(df["WorkingTime"], errors="coerce")
    else:
        df["WorkingTime"] = pd.NaT

    if df["WorkingTime"].isna().mean() > 0.5:
        df["WorkingTime"] = pd.date_range("2024-01-01 09:00:00", periods=len(df), freq="3s") + pd.to_timedelta(offset_seconds, unit="s")

    df["dataset"] = dataset_name
    return df


def fit_setpower_realpower_model(train_df, setpower_col="SetPower", realpower_col="RealPower", min_iqr=1e-6):
    train_df = train_df.copy()
    train_df[setpower_col] = pd.to_numeric(train_df[setpower_col], errors="coerce")
    train_df[realpower_col] = pd.to_numeric(train_df[realpower_col], errors="coerce")
    train_df = train_df.dropna(subset=[setpower_col, realpower_col])

    model_stat = {}
    for sp_value, group_df in train_df.groupby(setpower_col):
        values = group_df[realpower_col]
        median = values.median()
        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1
        if pd.isna(iqr) or iqr == 0:
            iqr = min_iqr

        model_stat[sp_value] = {
            "median": median,
            "iqr": iqr,
        }

    trained_setpowers = sorted(model_stat.keys())
    return {"setpower_col": setpower_col, "realpower_col": realpower_col, "model_stat": model_stat, "trained_setpowers": trained_setpowers}


def find_nearest_setpower(sp_value, trained_setpowers):
    if pd.isna(sp_value):
        return trained_setpowers[0]
    return min(trained_setpowers, key=lambda x: abs(x - sp_value))


def judge_setpower_realpower_improved(model, test_df, threshold=5.0):
    result_df = test_df.copy()

    setpower_col = model["setpower_col"]
    realpower_col = model["realpower_col"]
    model_stat = model["model_stat"]
    trained_setpowers = model["trained_setpowers"]

    scores = []
    preds = []
    lower_bounds = []
    upper_bounds = []
    power_up_needed = []
    power_down_needed = []

    for _, row in result_df.iterrows():
        sp = row.get(setpower_col, np.nan)
        rp = row.get(realpower_col, np.nan)

        if sp in model_stat:
            use_sp = sp
        else:
            use_sp = find_nearest_setpower(sp, trained_setpowers)

        stat = model_stat[use_sp]
        median = stat["median"]
        iqr = stat["iqr"]

        lower_bound = median - (threshold * iqr)
        upper_bound = median + (threshold * iqr)

        if pd.isna(rp):
            score = np.nan
            pred = 1
            up_needed = 0.0
            down_needed = 0.0
        else:
            score = abs(rp - median) / iqr
            pred = 1 if score > threshold else 0

            if rp < lower_bound:
                up_needed = float(lower_bound - rp)
                down_needed = 0.0
            elif rp > upper_bound:
                up_needed = 0.0
                down_needed = float(rp - upper_bound)
            else:
                up_needed = 0.0
                down_needed = 0.0

        scores.append(score)
        preds.append(pred)
        lower_bounds.append(lower_bound)
        upper_bounds.append(upper_bound)
        power_up_needed.append(up_needed)
        power_down_needed.append(down_needed)

    result_df["realpower_robust_score"] = scores
    result_df["pred_label"] = preds
    result_df["realpower_lower_bound"] = lower_bounds
    result_df["realpower_upper_bound"] = upper_bounds
    result_df["realpower_up_needed"] = power_up_needed
    result_df["realpower_down_needed"] = power_down_needed
    result_df["realpower_up_needed_text"] = result_df["realpower_up_needed"].fillna(0).round(0).astype(int).map(lambda x: f"▲ {x}")
    result_df["realpower_down_needed_text"] = result_df["realpower_down_needed"].fillna(0).round(0).astype(int).map(lambda x: f"▼ {x}")
    return result_df


def safe_label_array(label_array, target_len):
    label_array = np.asarray(label_array, dtype=int)
    if len(label_array) == target_len:
        return label_array
    if len(label_array) > target_len:
        return label_array[:target_len]
    result = np.zeros(target_len, dtype=int)
    result[:len(label_array)] = label_array
    return result


MODEL_BASE_FEATURES = ["PageNo", "Speed", "Length", "RealPower", "SetPower", "GateOnTime"]
MODEL_STEP1_FEATURES = ["Condition_Zscore", "Is_Unknown_Recipe", "Energy_per_Length"]
MODEL_STEP2_FEATURES = ["Is_Unknown_Recipe", "Condition_Zscore", "RealPower"]
MODEL_PCA_FEATURES = ["PageNo", "Speed", "Length", "RealPower", "SetPower", "GateOnTime", "Condition_Zscore"]
DEFECT_MODEL_OPTIONS = [
    "SetPower Robust-IQR",
    "Z-score",
    "IQR",
    "Isolation Forest",
    "One-Class SVM",
    "LOF",
    "Isolation Forest - Step1 Feature",
    "Isolation Forest - Step2 Rule",
    "Isolation Forest - Step3 Feature",
    "PCA IF",
    "PCA OCSVM",
    "PCA OR 앙상블",
    "PCA AND 앙상블",
    "Final Condition Zscore Rule",
]


def add_default_prediction_columns(result_df: pd.DataFrame) -> pd.DataFrame:
    result_df = result_df.copy()
    for col, default_value in [
        ("realpower_robust_score", np.nan),
        ("realpower_lower_bound", np.nan),
        ("realpower_upper_bound", np.nan),
        ("realpower_up_needed", 0.0),
        ("realpower_down_needed", 0.0),
        ("realpower_up_needed_text", ""),
        ("realpower_down_needed_text", ""),
    ]:
        if col not in result_df.columns:
            result_df[col] = default_value
    return result_df


def build_modeling_dataset(train_df, ok1_df, ok2_df, ng1_df, ng2_df) -> pd.DataFrame:
    frames = []
    for name, df, default_label in [
        ("TRAIN", train_df, 0),
        ("OK1", ok1_df, 0),
        ("OK2", ok2_df, 0),
        ("NG1", ng1_df, None),
        ("NG2", ng2_df, None),
    ]:
        temp = df.copy()
        temp["dataset"] = name
        if "true_label" in temp.columns:
            temp["true_label"] = pd.to_numeric(temp["true_label"], errors="coerce").fillna(0).astype(int)
        elif default_label is not None:
            temp["true_label"] = int(default_label)
        else:
            temp["true_label"] = 0
        frames.append(temp)

    all_df = pd.concat(frames, ignore_index=True)
    all_df["WorkingTime"] = pd.to_datetime(all_df["WorkingTime"], errors="coerce")
    all_df["_sort_time"] = all_df["WorkingTime"].fillna(pd.Timestamp.max)
    all_df = all_df.sort_values(["_sort_time", "dataset", "order"], kind="stable").reset_index(drop=True)

    page_values = pd.to_numeric(all_df.get("PageNo"), errors="coerce")
    group_start = page_values.eq(1)
    if len(group_start) > 0:
        group_start.iloc[0] = True
    all_df["model_group"] = group_start.cumsum().astype(int)
    all_df = all_df.drop(columns=["_sort_time"])
    return all_df


def summarize_model_groups(modeling_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group_id, group_df in modeling_df.groupby("model_group", sort=True):
        rows.append({
            "그룹": int(group_id),
            "데이터": ", ".join(sorted(group_df["dataset"].astype(str).unique())),
            "행수": int(len(group_df)),
            "시작시간": group_df["WorkingTime"].min(),
            "종료시간": group_df["WorkingTime"].max(),
            "불량수": int(pd.to_numeric(group_df["true_label"], errors="coerce").fillna(0).sum()),
        })
    return pd.DataFrame(rows)


def parse_group_selection(selection_text: str, valid_groups) -> tuple[list[int], list[str]]:
    valid_set = {int(v) for v in valid_groups}
    selected = set()
    invalid = []
    text = str(selection_text or "").strip()
    if not text:
        return [], []

    normalized = text.replace("~", "-").replace(" ", "")
    for token in normalized.split(","):
        if not token:
            continue
        try:
            if "-" in token:
                start_text, end_text = token.split("-", 1)
                start_value = int(start_text)
                end_value = int(end_text)
                if start_value > end_value:
                    start_value, end_value = end_value, start_value
                values = range(start_value, end_value + 1)
            else:
                values = [int(token)]
        except Exception:
            invalid.append(token)
            continue

        for value in values:
            if value in valid_set:
                selected.add(value)
            else:
                invalid.append(str(value))

    return sorted(selected), sorted(set(invalid), key=lambda x: int(x) if str(x).isdigit() else str(x))


def format_group_range_text(group_ids) -> str:
    ids = sorted({int(group_id) for group_id in group_ids})
    if not ids:
        return ""

    ranges = []
    start = ids[0]
    prev = ids[0]
    for group_id in ids[1:]:
        if group_id == prev + 1:
            prev = group_id
            continue
        ranges.append(f"{start}-{prev}" if start != prev else str(start))
        start = group_id
        prev = group_id

    ranges.append(f"{start}-{prev}" if start != prev else str(start))
    return ", ".join(ranges)


def get_defect_group_ids(summary_df: pd.DataFrame) -> list[int]:
    if summary_df is None or len(summary_df) == 0:
        return []
    if "그룹" not in summary_df.columns or "불량수" not in summary_df.columns:
        return []

    defect_rows = summary_df[pd.to_numeric(summary_df["불량수"], errors="coerce").fillna(0) > 0]
    return sorted(defect_rows["그룹"].astype(int).tolist())


def build_model_prediction_result(trained_model: dict, predict_df: pd.DataFrame, predict_group_ids: list[int]) -> dict:
    pred_df = predict_defect_model(trained_model, predict_df, fallback_threshold=threshold)
    y_true = pd.to_numeric(pred_df["true_label"], errors="coerce").fillna(0).astype(int).to_numpy()
    y_pred = pd.to_numeric(pred_df["pred_label"], errors="coerce").fillna(0).astype(int).to_numpy()
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    metric_df = pd.DataFrame([{
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
    }]).round(4)

    return {
        "groups": predict_group_ids,
        "cm": cm,
        "metrics": metric_df,
        "model_signature": trained_model.get("signature"),
    }


def append_model_training_history(trained_model: dict, train_group_text: str, prediction_result: dict) -> None:
    history_key = (
        f"{trained_model.get('signature')}|"
        f"predict={','.join(map(str, prediction_result.get('groups', [])))}"
    )
    history = st.session_state.setdefault("model_training_history", [])
    if any(row.get("_기록키") == history_key for row in history):
        return

    metric_df = prediction_result["metrics"]
    history.append({
        "_기록키": history_key,
        "모델명": trained_model.get("name", ""),
        "학습시킨 그룹범위": train_group_text,
        "recall": float(metric_df["Recall"].iloc[0]),
        "f1-score": float(metric_df["F1"].iloc[0]),
        "Accuracy": float(metric_df["Accuracy"].iloc[0]),
    })


def render_prediction_result(prediction_result: dict) -> None:
    cm = prediction_result["cm"]
    st.plotly_chart(make_confusion_matrix_figure(cm), use_container_width=True)
    st.dataframe(
        pd.DataFrame(cm, index=["실제 정상", "실제 불량"], columns=["예측 정상", "예측 불량"]),
        use_container_width=True,
    )
    st.dataframe(prediction_result["metrics"], use_container_width=True, hide_index=True)


def build_feature_metadata(train_df: pd.DataFrame) -> dict:
    numeric_train = train_df.copy()
    for col in ["Speed", "Length", "SetPower", "GateOnTime", "RealPower", "PageNo"]:
        if col in numeric_train.columns:
            numeric_train[col] = pd.to_numeric(numeric_train[col], errors="coerce")

    baseline_stats = (
        numeric_train
        .dropna(subset=["Speed", "Length", "GateOnTime", "RealPower"])
        .groupby(["Speed", "Length", "GateOnTime"])["RealPower"]
        .agg(Baseline_Mean="mean", Baseline_Std="std")
        .reset_index()
    )
    baseline_stats["Baseline_Std"] = (
        pd.to_numeric(baseline_stats["Baseline_Std"], errors="coerce")
        .fillna(0)
        .clip(lower=1e-6)
    )

    recipe_df = numeric_train[["Speed", "Length", "SetPower", "GateOnTime"]].dropna().drop_duplicates()
    normal_recipes = set(map(tuple, recipe_df.to_numpy()))
    return {"baseline_stats": baseline_stats, "normal_recipes": normal_recipes}


def add_engineered_features(df: pd.DataFrame, feature_meta: dict) -> pd.DataFrame:
    result = df.copy()
    for col in ["PageNo", "Speed", "Length", "RealPower", "SetPower", "GateOnTime"]:
        if col not in result.columns:
            result[col] = np.nan
        result[col] = pd.to_numeric(result[col], errors="coerce")

    setpower_w = result["SetPower"] * 20
    result["SetPower_W"] = setpower_w
    result["Power_Diff"] = result["RealPower"] - setpower_w
    result["Power_Diff_Ratio"] = result["Power_Diff"] / setpower_w.replace(0, np.nan)
    result["Energy_per_Length"] = (
        result["RealPower"] * result["GateOnTime"] / result["Length"].replace(0, np.nan)
    )

    drop_cols = ["Baseline_Mean", "Baseline_Std", "Condition_Error", "Condition_Zscore"]
    result = result.drop(columns=[c for c in drop_cols if c in result.columns], errors="ignore")
    result = result.merge(
        feature_meta["baseline_stats"],
        on=["Speed", "Length", "GateOnTime"],
        how="left",
    )
    result["Condition_Error"] = result["RealPower"] - result["Baseline_Mean"]
    result["Condition_Zscore"] = result["Condition_Error"] / result["Baseline_Std"].replace(0, np.nan)

    normal_recipes = feature_meta["normal_recipes"]
    result["Is_Unknown_Recipe"] = result.apply(
        lambda row: 0
        if (row["Speed"], row["Length"], row["SetPower"], row["GateOnTime"]) in normal_recipes
        else 1,
        axis=1,
    )
    return result


def numeric_feature_matrix(df: pd.DataFrame, features: list[str], fill_values=None):
    matrix_df = df.copy()
    for feature in features:
        if feature not in matrix_df.columns:
            matrix_df[feature] = np.nan
    matrix_df = matrix_df[features].apply(pd.to_numeric, errors="coerce")
    matrix_df = matrix_df.replace([np.inf, -np.inf], np.nan)
    if "Condition_Zscore" in matrix_df.columns:
        matrix_df["Condition_Zscore"] = matrix_df["Condition_Zscore"].fillna(999.0)

    if fill_values is None:
        fill_values = matrix_df.median(numeric_only=True).fillna(0.0)

    matrix_df = matrix_df.fillna(fill_values).fillna(0.0)
    return matrix_df, fill_values


def fit_defect_model(model_name: str, train_df: pd.DataFrame, robust_threshold: float = 5.0) -> dict:
    if len(train_df) < 2:
        raise ValueError("학습 데이터가 너무 적습니다. 최소 2행 이상 선택해주세요.")

    if model_name == "SetPower Robust-IQR":
        return {
            "name": model_name,
            "kind": "robust_iqr",
            "model": fit_setpower_realpower_model(train_df),
            "threshold": robust_threshold,
        }

    if model_name == "Final Condition Zscore Rule":
        feature_meta = build_feature_metadata(train_df)
        return {"name": model_name, "kind": "condition_rule", "feature_meta": feature_meta, "threshold": 10.0}

    feature_meta = build_feature_metadata(train_df)
    train_features_df = add_engineered_features(train_df, feature_meta)

    if model_name in ["Z-score", "IQR", "Isolation Forest", "One-Class SVM", "LOF"]:
        features = MODEL_BASE_FEATURES
    elif model_name == "Isolation Forest - Step1 Feature":
        features = MODEL_STEP1_FEATURES
    elif model_name in ["Isolation Forest - Step2 Rule", "Isolation Forest - Step3 Feature"]:
        features = MODEL_STEP2_FEATURES
    else:
        features = MODEL_PCA_FEATURES

    X_train, fill_values = numeric_feature_matrix(train_features_df, features)

    if model_name == "Z-score":
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
        std = X_scaled.std(axis=0)
        std[std == 0] = 1e-6
        return {"name": model_name, "kind": "zscore", "features": features, "feature_meta": feature_meta, "fill_values": fill_values, "scaler": scaler, "mean": X_scaled.mean(axis=0), "std": std, "threshold": 3.0}

    if model_name == "IQR":
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
        q1 = np.percentile(X_scaled, 25, axis=0)
        q3 = np.percentile(X_scaled, 75, axis=0)
        iqr = q3 - q1
        iqr[iqr == 0] = 1e-6
        return {"name": model_name, "kind": "iqr", "features": features, "feature_meta": feature_meta, "fill_values": fill_values, "scaler": scaler, "q1": q1, "q3": q3, "iqr": iqr, "factor": 1.5}

    if model_name.startswith("PCA "):
        scaler = StandardScaler()
        Z_train = scaler.fit_transform(X_train)
        pca = PCA(n_components=0.95, random_state=42)
        P_train = pca.fit_transform(Z_train)
        iso = IsolationForest(n_estimators=300, random_state=42).fit(P_train)
        sample_count = min(len(P_train), 15000)
        sample_idx = np.random.RandomState(42).choice(len(P_train), sample_count, replace=False)
        ocsvm = OneClassSVM(kernel="rbf", gamma="scale", nu=0.001).fit(P_train[sample_idx])
        alpha = 0.001
        return {
            "name": model_name,
            "kind": "pca_ensemble",
            "features": features,
            "feature_meta": feature_meta,
            "fill_values": fill_values,
            "scaler": scaler,
            "pca": pca,
            "iso": iso,
            "ocsvm": ocsvm,
            "thr_iso": float(np.quantile(iso.score_samples(P_train), alpha)),
            "thr_oc": float(np.quantile(ocsvm.decision_function(P_train), alpha)),
            "combine": "or" if "OR" in model_name else "and" if "AND" in model_name else "if" if model_name == "PCA IF" else "ocsvm",
        }

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    if model_name == "One-Class SVM":
        sample_count = min(len(X_scaled), 15000)
        sample_idx = np.random.RandomState(42).choice(len(X_scaled), sample_count, replace=False)
        estimator = OneClassSVM(kernel="rbf", gamma="scale", nu=0.01).fit(X_scaled[sample_idx])
    elif model_name == "LOF":
        n_neighbors = max(1, min(20, len(X_scaled) - 1))
        estimator = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True).fit(X_scaled)
    else:
        contamination = 0.005 if model_name == "Isolation Forest - Step3 Feature" else 0.01
        estimator = IsolationForest(contamination=contamination, random_state=42).fit(X_scaled)

    return {
        "name": model_name,
        "kind": "sklearn_outlier",
        "features": features,
        "feature_meta": feature_meta,
        "fill_values": fill_values,
        "scaler": scaler,
        "estimator": estimator,
    }


def predict_defect_model(fitted_model: dict, df: pd.DataFrame, fallback_threshold: float = 5.0) -> pd.DataFrame:
    if fitted_model["kind"] == "robust_iqr":
        model_threshold = fitted_model.get("threshold", fallback_threshold)
        return judge_setpower_realpower_improved(fitted_model["model"], df, threshold=model_threshold)

    result_df = df.copy()

    if fitted_model["kind"] == "condition_rule":
        feature_df = add_engineered_features(result_df, fitted_model["feature_meta"])
        z_values = pd.to_numeric(feature_df["Condition_Zscore"], errors="coerce")
        pred = ((z_values.abs() > fitted_model.get("threshold", 10.0)) | z_values.isna()).astype(int).to_numpy()
        result_df["model_score"] = z_values.abs()
    else:
        feature_df = add_engineered_features(result_df, fitted_model["feature_meta"])
        X, _ = numeric_feature_matrix(feature_df, fitted_model["features"], fitted_model["fill_values"])

        if fitted_model["kind"] == "zscore":
            X_scaled = fitted_model["scaler"].transform(X)
            z = np.abs((X_scaled - fitted_model["mean"]) / fitted_model["std"])
            pred = (z.max(axis=1) > fitted_model["threshold"]).astype(int)
            result_df["model_score"] = z.max(axis=1)
        elif fitted_model["kind"] == "iqr":
            X_scaled = fitted_model["scaler"].transform(X)
            lower = fitted_model["q1"] - fitted_model["factor"] * fitted_model["iqr"]
            upper = fitted_model["q3"] + fitted_model["factor"] * fitted_model["iqr"]
            pred = ((X_scaled < lower) | (X_scaled > upper)).any(axis=1).astype(int)
            result_df["model_score"] = np.maximum(np.maximum(lower - X_scaled, X_scaled - upper), 0).max(axis=1)
        elif fitted_model["kind"] == "pca_ensemble":
            P = fitted_model["pca"].transform(fitted_model["scaler"].transform(X))
            pred_if = (fitted_model["iso"].score_samples(P) < fitted_model["thr_iso"]).astype(int)
            pred_oc = (fitted_model["ocsvm"].decision_function(P) < fitted_model["thr_oc"]).astype(int)
            combine = fitted_model["combine"]
            if combine == "or":
                pred = ((pred_if == 1) | (pred_oc == 1)).astype(int)
            elif combine == "and":
                pred = ((pred_if == 1) & (pred_oc == 1)).astype(int)
            elif combine == "ocsvm":
                pred = pred_oc
            else:
                pred = pred_if
            result_df["model_score"] = np.minimum(
                fitted_model["iso"].score_samples(P) - fitted_model["thr_iso"],
                fitted_model["ocsvm"].decision_function(P) - fitted_model["thr_oc"],
            )
        else:
            X_scaled = fitted_model["scaler"].transform(X)
            pred = (fitted_model["estimator"].predict(X_scaled) == -1).astype(int)
            if hasattr(fitted_model["estimator"], "decision_function"):
                result_df["model_score"] = fitted_model["estimator"].decision_function(X_scaled)

    result_df["pred_label"] = np.asarray(pred, dtype=int)
    return add_default_prediction_columns(result_df)


def make_confusion_matrix_figure(cm: np.ndarray):
    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=["예측 정상", "예측 불량"],
            y=["실제 정상", "실제 불량"],
            colorscale="Blues",
            showscale=False,
            text=cm,
            texttemplate="%{text}",
            textfont=dict(size=18, color="#111827"),
        )
    )
    fig.update_layout(height=330, margin=dict(l=20, r=20, t=20, b=20))
    return fig




def get_current_day_df(seen_df: pd.DataFrame, latest_row: pd.Series) -> pd.DataFrame:
    """
    현재 row의 WorkingTime 날짜를 기준으로 현재 날짜 데이터만 반환합니다.

    목적:
    - 일별 생산량 / 일별 불량은 누적 전체가 아니라 현재 날짜 기준으로 계산
    - WorkingTime 날짜가 바뀌면 이전 날짜 데이터는 제외되므로 자동으로 0부터 다시 누적됨
    """
    if seen_df is None or len(seen_df) == 0 or "WorkingTime" not in seen_df.columns:
        return pd.DataFrame()

    latest_time = pd.to_datetime(latest_row.get("WorkingTime"), errors="coerce")

    if pd.isna(latest_time):
        return seen_df.copy()

    current_date = latest_time.date()

    temp_df = seen_df.copy()
    temp_df["_work_date"] = pd.to_datetime(temp_df["WorkingTime"], errors="coerce").dt.date

    return temp_df[temp_df["_work_date"] == current_date].copy()


def count_completed_products(df: pd.DataFrame, page_col: str = "PageNo", complete_page_no: int = 39) -> int:
    """
    생산 완료 수량 계산 함수.
    PageNo가 39인 행이 나타날 때 제품 1개 생산 완료로 계산합니다.
    """
    if df is None or len(df) == 0 or page_col not in df.columns:
        return 0

    page_values = pd.to_numeric(df[page_col], errors="coerce")
    return int((page_values == complete_page_no).sum())



def get_workingtime_delay_seconds(
    df: pd.DataFrame,
    current_idx: int,
    default_seconds: float = 0.8,
    cap_threshold_seconds: float = 10.0,
    capped_seconds: float = 5.0,
) -> float:
    """
    현재 row와 다음 row의 WorkingTime 차이를 계산하여 다음 row 출력까지의 대기 시간을 반환합니다.

    규칙:
    - 다음 row와 현재 row의 WorkingTime 차이를 계산
    - 차이가 10초 이상이면 5초로 고정
    - 차이가 10초 미만이면 실제 WorkingTime 간격 그대로 사용
    - WorkingTime이 없거나 비정상 값이면 default_seconds 사용
    """
    if df is None or len(df) == 0 or "WorkingTime" not in df.columns:
        return float(default_seconds)

    # current_idx는 현재까지 표시된 row 수
    # 마지막으로 표시된 row 위치 = current_idx - 1
    current_pos = max(0, min(int(current_idx) - 1, len(df) - 1))
    next_pos = current_pos + 1

    if next_pos >= len(df):
        return float(default_seconds)

    current_time = pd.to_datetime(df.iloc[current_pos]["WorkingTime"], errors="coerce")
    next_time = pd.to_datetime(df.iloc[next_pos]["WorkingTime"], errors="coerce")

    if pd.isna(current_time) or pd.isna(next_time):
        return float(default_seconds)

    diff_seconds = (next_time - current_time).total_seconds()

    if diff_seconds <= 0 or not np.isfinite(diff_seconds):
        return float(default_seconds)

    if diff_seconds >= cap_threshold_seconds:
        return float(capped_seconds)

    return float(diff_seconds)


MIN_MONITOR_RENDER_SECONDS = 0.45
DEFECT_FLASH_COOLDOWN_SECONDS = 2.0
LOG_HIDDEN_COLUMNS = [
    "dataset",
    "order",
    "cycle_id",
    "PageNo",
    "real_power_robust",
    "realpower_robust",
    "realpower_robust_score",
]
RECIPE_NORMAL_LABEL = "\uc815\uc0c1"
RECIPE_DEFECT_LABEL = "\ubd88\ub7c9"
RECIPE_TOTAL_LABEL = "\ud569\uacc4"
RECIPE_RATE_LABEL = "\ube44\uc728"
RECIPE_FIELD_COLUMNS = ["Speed", "Length", "SetPower", "SetFrequency", "SetDuty", "GateOnTime", "RealPower"]


def estimate_monitor_refresh_seconds(
    df: pd.DataFrame,
    use_workingtime_interval: bool,
    fixed_seconds: float,
    max_interval_seconds: float,
    min_seconds: float = MIN_MONITOR_RENDER_SECONDS,
) -> float:
    """
    Fragment가 같은 화면을 너무 자주 다시 그리지 않도록 실제 재생 간격에 가까운 갱신 주기를 계산합니다.
    """
    min_seconds = float(min_seconds)
    fixed_seconds = float(fixed_seconds)
    max_interval_seconds = float(max_interval_seconds)

    if not use_workingtime_interval:
        return max(min_seconds, fixed_seconds)

    if df is None or len(df) < 2 or "WorkingTime" not in df.columns:
        return max(min_seconds, fixed_seconds)

    times = pd.to_datetime(df["WorkingTime"], errors="coerce")
    diffs = times.diff().dt.total_seconds().dropna()

    if len(diffs) == 0:
        return max(min_seconds, fixed_seconds)

    diff_values = diffs.to_numpy(dtype=float)
    diff_values = diff_values[np.isfinite(diff_values) & (diff_values > 0)]

    if diff_values.size == 0:
        estimated_seconds = fixed_seconds
    else:
        estimated_seconds = float(np.median(np.minimum(diff_values, max_interval_seconds)))

    return min(max_interval_seconds, max(min_seconds, estimated_seconds))



def format_realpower_with_adjustment(row) -> str:
    """
    로그 표시용 RealPower 문자열 생성 함수.

    별도 상승/하강 필요량 컬럼을 만들지 않고,
    RealPower 열 안에 정상 복귀 필요량을 함께 표시합니다.

    예:
    - 정상: 1688
    - 낮아서 불량: 701 (▲ 193)
    - 높아서 불량: 2100 (▼ 85)
    """
    rp = row.get("RealPower", np.nan)
    up_needed = row.get("realpower_up_needed", 0)
    down_needed = row.get("realpower_down_needed", 0)

    try:
        rp_text = f"{float(rp):.0f}"
    except Exception:
        rp_text = str(rp)

    try:
        up_value = float(up_needed)
    except Exception:
        up_value = 0.0

    try:
        down_value = float(down_needed)
    except Exception:
        down_value = 0.0

    if up_value > 0:
        return f"{rp_text} (▲ {up_value:.0f})"

    if down_value > 0:
        return f"{rp_text} (▼ {down_value:.0f})"

    return rp_text


def highlight_defect_rows(row):
    """
    st.dataframe에서 불량 행 전체를 연한 빨간색으로 표시하고,
    RealPower 열 안의 상승/하강 필요량을 색상으로 강조하기 위한 함수
    """
    pred_value = row.get("pred_label", "")
    is_defect = pred_value == "불량" or pred_value == 1

    styles = []

    for col in row.index:
        cell_style = "font-family: Arial, Helvetica, sans-serif; font-weight: 500; color: #111827;"

        if is_defect:
            cell_style += "background-color: #ffe4e6;"

        if col == "RealPower":
            realpower_text = str(row.get("RealPower", ""))
            if "▲" in realpower_text:
                cell_style += " color: #16a34a; font-weight: 700;"
            elif "▼" in realpower_text:
                cell_style += " color: #dc2626; font-weight: 700;"

        elif col == "pred_label":
            if is_defect:
                cell_style += " color: #991b1b; font-weight: 700;"
            else:
                cell_style += " color: #1d4ed8; font-weight: 700;"

        styles.append(cell_style)

    return styles


def format_recipe_group_value(value, decimals: int = 1) -> str:
    if pd.isna(value):
        return ""

    try:
        number = float(value)
    except Exception:
        return str(value)

    if abs(number - round(number)) < 1e-9:
        return f"{number:.0f}"

    return f"{number:.{decimals}f}"


def is_defect_label_value(value) -> bool:
    if pd.isna(value):
        return False

    if isinstance(value, (bool, np.bool_)):
        return bool(value)

    try:
        numeric_value = float(value)
        if np.isfinite(numeric_value):
            return int(round(numeric_value)) == 1
    except Exception:
        pass

    label_text = str(value).strip().lower()
    if label_text in {"1", "true", "yes", "y"}:
        return True

    defect_tokens = [
        RECIPE_DEFECT_LABEL,
        "ng",
        "defect",
        "bad",
        "fail",
        "failed",
        "abnormal",
        "error",
        "alarm",
        "遺덈웾",
    ]
    normal_tokens = [
        RECIPE_NORMAL_LABEL,
        "ok",
        "normal",
        "good",
        "?뺤긽",
    ]

    if any(token in label_text for token in defect_tokens):
        return True

    if any(token in label_text for token in normal_tokens):
        return False

    return False


def build_recipe_combination_count_df(df: pd.DataFrame) -> pd.DataFrame:
    recipe_cols = [c for c in RECIPE_FIELD_COLUMNS if c in df.columns]

    if df is None or len(df) == 0 or not recipe_cols or "pred_label" not in df.columns:
        return pd.DataFrame()

    recipe_df = df[recipe_cols + ["pred_label"]].copy()
    round_digits = {
        "Speed": 1,
        "Length": 1,
        "SetPower": 0,
        "SetFrequency": 0,
        "SetDuty": 0,
        "GateOnTime": 0,
        "RealPower": 0,
    }

    for col in recipe_cols:
        numeric_values = pd.to_numeric(recipe_df[col], errors="coerce")
        if numeric_values.notna().any():
            digits = round_digits.get(col, 1)
            recipe_df[col] = numeric_values.round(digits).map(
                lambda value, digits=digits: format_recipe_group_value(value, digits)
            )
        else:
            recipe_df[col] = recipe_df[col].fillna("").astype(str)

    defect_mask = recipe_df["pred_label"].map(is_defect_label_value)
    recipe_df["_result"] = np.where(defect_mask, RECIPE_DEFECT_LABEL, RECIPE_NORMAL_LABEL)

    count_df = (
        recipe_df.groupby(recipe_cols + ["_result"], dropna=False)
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    for result_col in [RECIPE_NORMAL_LABEL, RECIPE_DEFECT_LABEL]:
        if result_col not in count_df.columns:
            count_df[result_col] = 0

    count_df[RECIPE_TOTAL_LABEL] = count_df[RECIPE_NORMAL_LABEL] + count_df[RECIPE_DEFECT_LABEL]
    count_df[RECIPE_RATE_LABEL] = count_df[RECIPE_TOTAL_LABEL] / max(len(df), 1)

    return count_df[recipe_cols + [RECIPE_NORMAL_LABEL, RECIPE_DEFECT_LABEL, RECIPE_TOTAL_LABEL, RECIPE_RATE_LABEL]].sort_values(
        [RECIPE_TOTAL_LABEL, RECIPE_DEFECT_LABEL, RECIPE_NORMAL_LABEL],
        ascending=[False, False, False],
    )


def make_recipe_count_bar_cell(value, max_count: int, color: str, title: str = "") -> str:
    count_value = int(value) if pd.notna(value) else 0
    width_pct = 0 if max_count <= 0 else min(100, max(0, count_value / max_count * 100))
    title_attr = f' title="{html.escape(title, quote=True)}"' if title else ""

    return f"""
        <div class="recipe-count-cell"{title_attr}>
            <div class="recipe-count-bar" style="width: {width_pct:.1f}%; background: {color};"></div>
            <span>{count_value:,}</span>
        </div>
    """


def make_recipe_combination_count_html(count_df: pd.DataFrame) -> str:
    if count_df is None or len(count_df) == 0:
        return ""

    visible_df = count_df.copy()
    recipe_cols = [
        c for c in RECIPE_FIELD_COLUMNS
        if c in visible_df.columns
    ]
    max_count = max(int(visible_df[RECIPE_TOTAL_LABEL].max()), 1)

    header_cells = "".join(
        f"<th>{html.escape(str(col))}</th>"
        for col in recipe_cols + [RECIPE_TOTAL_LABEL]
    )

    body_rows = []
    for _, row in visible_df.iterrows():
        recipe_cells = "".join(
            f"<td>{html.escape(str(row.get(col, '')))}</td>"
            for col in recipe_cols
        )
        normal_count = int(row.get(RECIPE_NORMAL_LABEL, 0)) if pd.notna(row.get(RECIPE_NORMAL_LABEL, 0)) else 0
        defect_count = int(row.get(RECIPE_DEFECT_LABEL, 0)) if pd.notna(row.get(RECIPE_DEFECT_LABEL, 0)) else 0
        total_count = int(row.get(RECIPE_TOTAL_LABEL, 0)) if pd.notna(row.get(RECIPE_TOTAL_LABEL, 0)) else 0
        bar_color = "#ef4444" if defect_count > 0 else "#2563eb"
        bar_title = (
            f"{RECIPE_NORMAL_LABEL} {normal_count:,} / "
            f"{RECIPE_DEFECT_LABEL} {defect_count:,}"
        )
        total_cell = make_recipe_count_bar_cell(total_count, max_count, bar_color, bar_title)

        body_rows.append(
            "<tr>"
            + recipe_cells
            + f"<td>{total_cell}</td>"
            + "</tr>"
        )

    return f"""
    <style>
    .recipe-table-wrap {{
        height: 520px;
        overflow: auto;
        border: 1px solid #d0d5dd;
        border-radius: 8px;
        background: #ffffff;
    }}
    .recipe-table {{
        width: 100%;
        min-width: 860px;
        border-collapse: collapse;
        font-family: Arial, Helvetica, sans-serif;
        font-size: 13px;
        color: #111827;
    }}
    .recipe-table th {{
        position: sticky;
        top: 0;
        z-index: 1;
        background: #f8fafc;
        color: #111827;
        font-weight: 700;
        text-align: left;
        border-bottom: 1px solid #d0d5dd;
        padding: 8px 10px;
        white-space: nowrap;
    }}
    .recipe-table td {{
        border-bottom: 1px solid #eef2f6;
        padding: 7px 10px;
        white-space: nowrap;
        vertical-align: middle;
        font-weight: 500;
    }}
    .recipe-count-cell {{
        position: relative;
        min-width: 110px;
        height: 24px;
        border-radius: 3px;
        background: #f2f4f7;
        overflow: hidden;
    }}
    .recipe-count-bar {{
        position: absolute;
        inset: 0 auto 0 0;
        opacity: 0.88;
    }}
    .recipe-count-cell span {{
        position: relative;
        z-index: 1;
        display: block;
        padding: 3px 7px;
        color: #111827;
        font-weight: 700;
        font-family: Arial, Helvetica, sans-serif;
        line-height: 18px;
        text-align: right;
    }}
    .recipe-number-cell {{
        text-align: right;
        font-variant-numeric: tabular-nums;
    }}
    </style>
    <div class="recipe-table-wrap">
        <table class="recipe-table">
            <thead><tr>{header_cells}</tr></thead>
            <tbody>{''.join(body_rows)}</tbody>
        </table>
    </div>
    """


def make_chart(df_window, feature, latest_row, alarm=False):
    fig = go.Figure()

    # 전체 라인은 정상/불량 구분 없이 하나의 시계열로 자연스럽게 이어지게 표시
    all_df = df_window.copy()
    defect_df = df_window[df_window["pred_label"] == 1]

    fig.add_trace(
        go.Scatter(
            x=all_df["WorkingTime"],
            y=all_df[feature],
            mode="lines",
            name="정상",
            line=dict(color="#2563eb", width=2),
            hovertemplate=f"시간=%{{x}}<br>{feature}=%{{y}}<extra></extra>",
        )
    )

    # 불량은 라인을 따로 잇지 않고, 해당 지점만 빨간 점으로 강조
    if len(defect_df) > 0:
        fig.add_trace(
            go.Scatter(
                x=defect_df["WorkingTime"],
                y=defect_df[feature],
                mode="markers",
                name="불량",
                marker=dict(size=9, color="#ef4444"),
                hovertemplate=f"시간=%{{x}}<br>{feature}=%{{y}}<br>불량<extra></extra>",
            )
        )

    if latest_row is not None:
        latest_color = "#ef4444" if int(latest_row["pred_label"]) == 1 else "#10b981"
        fig.add_trace(
            go.Scatter(
                x=[latest_row["WorkingTime"]],
                y=[latest_row[feature]],
                mode="markers",
                name="현재",
                marker=dict(size=14, color=latest_color, line=dict(color="#111827", width=2)),
                hovertemplate=f"현재<br>시간=%{{x}}<br>{feature}=%{{y}}<extra></extra>",
            )
        )

    bg = "#fffafa" if alarm else "#ffffff"

    fig.update_layout(
        title=dict(text=f"WorkingTime - {feature}", x=0.02, xanchor="left"),
        height=300,
        margin=dict(l=30, r=20, t=42, b=30),
        paper_bgcolor=bg,
        plot_bgcolor=bg,
        xaxis_title="WorkingTime",
        yaxis_title=feature,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.4),
        uirevision=f"{feature}-stable",
        transition=dict(duration=0),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#eef2f6")
    fig.update_yaxes(showgrid=True, gridcolor="#eef2f6")
    return fig


# =========================================================
# 2. 데이터 준비
# =========================================================
train_raw, ok1_raw, ok2_raw, ng1_raw, ng2_raw, y_ng1, y_ng2, paths, missing = load_all_data()

train = prepare_time_and_order(train_raw, "TRAIN", 0)
ok1 = prepare_time_and_order(ok1_raw, "OK1", 0)
ok2 = prepare_time_and_order(ok2_raw, "OK2", len(ok1) * 3)
ng1 = prepare_time_and_order(ng1_raw, "NG1", (len(ok1) + len(ok2)) * 3)
ng2 = prepare_time_and_order(ng2_raw, "NG2", (len(ok1) + len(ok2) + len(ng1)) * 3)

ok1["true_label"] = 0
ok2["true_label"] = 0
ng1["true_label"] = safe_label_array(y_ng1, len(ng1))
ng2["true_label"] = safe_label_array(y_ng2, len(ng2))

test_all_base = pd.concat([ok1, ok2, ng1, ng2], ignore_index=True)
test_all_base["stream_order"] = np.arange(1, len(test_all_base) + 1)
modeling_all_base = build_modeling_dataset(train, ok1, ok2, ng1, ng2)
model_group_summary = summarize_model_groups(modeling_all_base)

for col in ["Speed", "RealPower", "SetPower", "SetFrequency", "SetDuty", "GateOnTime", "Length"]:
    if col not in test_all_base.columns:
        test_all_base[col] = np.nan
    if col not in train.columns:
        train[col] = np.nan


# =========================================================
# 3. 사이드바
# =========================================================
st.sidebar.title("설정")

threshold = st.sidebar.slider("불량 판정 임계값", 1.0, 12.0, 5.0, 0.5)
window_size = st.sidebar.slider("최근 표시 데이터 개수", 30, 800, 180, 10)

use_workingtime_interval = st.sidebar.checkbox(
    "WorkingTime 간격 사용",
    value=True,
    help="체크하면 현재 row와 다음 row의 WorkingTime 차이만큼 기다렸다가 다음 row를 표시합니다."
)

max_interval_seconds = st.sidebar.number_input(
    "10초 이상 간격 대체값(초)",
    min_value=1.0,
    max_value=30.0,
    value=5.0,
    step=0.5,
    help="WorkingTime 간격이 10초 이상이면 이 값으로 고정합니다."
)

batch_size = st.sidebar.slider(
    "한 번에 진행할 row 수",
    1,
    30,
    1,
    1,
    help="WorkingTime 간격 사용 시에는 기본적으로 1 row씩 진행하는 것을 권장합니다."
)

sleep_sec = st.sidebar.slider(
    "고정 재생 간격(초)",
    0.02,
    5.00,
    0.80,
    0.01,
    help="WorkingTime 간격 사용 옵션을 끄면 이 간격으로 재생됩니다."
)

target_qty = st.sidebar.number_input(
    "목표 생산량",
    min_value=1,
    max_value=1_000_000,
    value=1000,
    step=50,
    help="진행률은 PageNo가 39인 행의 개수 / 목표 생산량 기준으로 계산됩니다."
)

available_features = [
    col
    for col in ["RealPower", "SetPower", "SetFrequency", "SetDuty", "Speed", "GateOnTime", "Length"]
    if col in test_all_base.columns
]
selected_features = st.sidebar.multiselect(
    "표시할 y축 그래프",
    options=available_features,
    default=[col for col in ["RealPower", "SetPower", "SetFrequency", "SetDuty", "Speed"] if col in available_features],
)

if not selected_features:
    selected_features = ["RealPower"]

# 선택한 y축 그래프를 모두 표시합니다.
# 기존에는 selected_features[:4]로 4개까지만 표시했지만,
# 이제 5개를 모두 선택해도 전부 표시됩니다.

dataset_filter = st.sidebar.multiselect(
    "데이터셋 선택",
    ["OK1", "OK2", "NG1", "NG2"],
    default=["OK1", "OK2", "NG1", "NG2"],
)

st.sidebar.caption("파일 인식 상태")
for key, value in paths.items():
    if value is not None:
        st.sidebar.write(f"✅ {key}: {Path(value).name}")
    else:
        st.sidebar.write(f"⚠️ {key}: 파일 없음")

if missing:
    st.sidebar.warning("일부 CSV가 없어 예시 데이터로 표시 중입니다.")


# =========================================================
# 4. 모델 예측
# =========================================================
model = fit_setpower_realpower_model(train)
test_all = test_all_base[test_all_base["dataset"].isin(dataset_filter)].copy()
active_defect_model = st.session_state.get("active_defect_model")

if active_defect_model is None:
    test_all = judge_setpower_realpower_improved(model, test_all, threshold=threshold)
    active_model_name = "SetPower Robust-IQR"
else:
    try:
        test_all = predict_defect_model(active_defect_model, test_all, fallback_threshold=threshold)
        active_model_name = active_defect_model.get("name", "사용자 적용 모델")
    except Exception as exc:
        st.sidebar.error(f"적용 모델 예측 실패: {exc}")
        test_all = judge_setpower_realpower_improved(model, test_all, threshold=threshold)
        active_model_name = "SetPower Robust-IQR"

test_all = test_all.reset_index(drop=True)
test_all["stream_order"] = np.arange(1, len(test_all) + 1)

if "idx_light" not in st.session_state:
    st.session_state.idx_light = 0
if "is_running_light" not in st.session_state:
    st.session_state.is_running_light = False

if "last_flash_idx" not in st.session_state:
    st.session_state.last_flash_idx = -1
if "last_flash_time" not in st.session_state:
    st.session_state.last_flash_time = 0.0
if "last_defect_event_key" not in st.session_state:
    st.session_state.last_defect_event_key = None

if "prev_idx_light" not in st.session_state:
    st.session_state.prev_idx_light = 0

if "flash_event_id" not in st.session_state:
    st.session_state.flash_event_id = 0

if "next_update_time_light" not in st.session_state:
    st.session_state.next_update_time_light = time.time()

if st.session_state.idx_light > len(test_all):
    st.session_state.idx_light = len(test_all)

active_model_signature = active_model_name
if active_defect_model is not None:
    active_model_signature = active_defect_model.get("signature", active_model_name)

data_signature_light = (tuple(dataset_filter), float(threshold), len(test_all), active_model_signature)
if "data_signature_light" not in st.session_state:
    st.session_state.data_signature_light = data_signature_light
elif st.session_state.data_signature_light != data_signature_light:
    st.session_state.idx_light = min(st.session_state.idx_light, len(test_all))
    st.session_state.prev_idx_light = st.session_state.idx_light
    st.session_state.last_flash_idx = -1
    st.session_state.last_flash_time = 0.0
    st.session_state.last_defect_event_key = None
    st.session_state.data_signature_light = data_signature_light

st.sidebar.caption(f"적용 모델: {active_model_name}")


# =========================================================
# 5. 헤더
# =========================================================
st.markdown('<div class="title"><span style="vertical-align:middle;">⚡</span> 배터리팩 용접 공정 실시간 불량 탐지 대시보드</div>', unsafe_allow_html=True)

btn1, btn2, btn3, btn4, spacer = st.columns([1, 1, 1, 1, 3])
with btn1:
    start_clicked = st.button("▶ 재생", use_container_width=True)
with btn2:
    stop_clicked = st.button("■ 정지", use_container_width=True)
with btn3:
    step_clicked = st.button("⏭ 한 번 진행", use_container_width=True)
with btn4:
    reset_clicked = st.button("↩ 리셋", use_container_width=True)

if reset_clicked:
    st.session_state.idx_light = 0
    st.session_state.is_running_light = False
    st.session_state.last_flash_idx = -1
    st.session_state.last_flash_time = 0.0
    st.session_state.last_defect_event_key = None
    st.session_state.prev_idx_light = 0
    st.session_state.flash_event_id = 0
    st.session_state.next_update_time_light = time.time()

if stop_clicked:
    st.session_state.is_running_light = False

if step_clicked:
    st.session_state.is_running_light = False
    st.session_state.prev_idx_light = st.session_state.idx_light
    step_size = 1 if use_workingtime_interval else batch_size
    st.session_state.idx_light = min(len(test_all), st.session_state.idx_light + step_size)
    st.session_state.next_update_time_light = time.time()

if start_clicked:
    st.session_state.is_running_light = True
    st.session_state.next_update_time_light = time.time()


monitor_refresh_seconds = estimate_monitor_refresh_seconds(
    test_all,
    use_workingtime_interval=use_workingtime_interval,
    fixed_seconds=sleep_sec,
    max_interval_seconds=max_interval_seconds,
)
monitor_run_every = monitor_refresh_seconds if st.session_state.is_running_light else None
daily_log_run_every = max(3.0, monitor_refresh_seconds * 2) if st.session_state.is_running_light else None

components.html(
    make_scroll_position_guard_html(0, st.session_state.is_running_light),
    height=0,
    scrolling=False,
)


# =========================================================
# Fragment 사용 시 주의
# ---------------------------------------------------------
# Streamlit fragment 안에서는 fragment 밖에서 만든 st.empty(),
# tab container, placeholder 같은 외부 container에 widget을 쓰면
# StreamlitFragmentWidgetsNotAllowedOutsideError가 발생할 수 있다.
#
# 그래서 이 버전은:
# 1) fragment 밖에서 placeholder를 만들지 않고
# 2) fragment 내부에서 화면 요소를 직접 그림
# 3) 실시간 모니터링 fragment와 일별 로그 fragment를 분리함
# =========================================================

if st.session_state.idx_light == 0:
    initial_step = 1 if use_workingtime_interval else batch_size
    st.session_state.idx_light = min(initial_step, len(test_all))


def get_current_frames(current_idx: int):
    """
    현재 진행 위치 기준으로 필요한 데이터프레임을 반환합니다.
    """
    current_idx = max(1, min(current_idx, len(test_all)))
    seen_df = test_all.iloc[:current_idx].copy()
    latest_row = seen_df.iloc[-1]

    start_idx = max(0, current_idx - window_size)
    window_df = test_all.iloc[start_idx:current_idx].copy()

    return current_idx, seen_df, latest_row, window_df


def render_monitor_content(current_idx: int):
    """
    실시간 모니터링 탭 내용.
    fragment 내부에서 직접 호출되어야 합니다.
    """
    if len(test_all) == 0:
        st.error("표시할 데이터가 없습니다.")
        return

    current_idx, seen_df, latest_row, window_df = get_current_frames(current_idx)

    latest_alarm = int(latest_row["pred_label"]) == 1
    normal_count = int((seen_df["pred_label"] == 0).sum())
    defect_count = int((seen_df["pred_label"] == 1).sum())

    # -----------------------------------------------------
    # 새로 처리된 구간에서 불량 발생 여부 확인
    # -----------------------------------------------------
    prev_idx = int(st.session_state.get("prev_idx_light", 0))
    prev_idx = max(0, min(prev_idx, current_idx))

    new_rows = test_all.iloc[prev_idx:current_idx]
    new_defect_rows = new_rows[new_rows["pred_label"] == 1]
    new_defect_detected = len(new_defect_rows) > 0

    # 불량이 새로 발생할 때만 flash하되, 연속 불량에서는 과도한 overlay 생성을 막습니다.
    if new_defect_detected:
        if "stream_order" in new_defect_rows.columns:
            defect_event_key = int(new_defect_rows["stream_order"].iloc[-1])
        else:
            defect_event_key = current_idx

        now = time.time()
        last_flash_time = float(st.session_state.get("last_flash_time", 0.0))
        is_new_event = st.session_state.get("last_defect_event_key") != defect_event_key
        can_flash = (now - last_flash_time) >= DEFECT_FLASH_COOLDOWN_SECONDS

        if is_new_event and can_flash:
            st.session_state.flash_event_id += 1
            components.html(
                make_flash_overlay_html(st.session_state.flash_event_id),
                height=0,
                scrolling=False,
            )
            st.session_state.last_flash_idx = current_idx
            st.session_state.last_flash_time = now

        if is_new_event:
            st.session_state.last_defect_event_key = defect_event_key

    st.session_state.prev_idx_light = current_idx

    # -----------------------------------------------------
    # 상태 배너
    # -----------------------------------------------------
    if latest_alarm:
            st.markdown(
                f'<div class="state-banner-alert">🚨 불량 감지 | 시간: {latest_row["WorkingTime"]}</div>',
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            '<div class="state-banner-normal">✅ 현재 공정 상태 정상 모니터링 중</div>',
            unsafe_allow_html=True,
        )

    # -----------------------------------------------------
    # 정상 / 불량 누적 카운터
    # -----------------------------------------------------
    st.markdown(
        f"""
        <div class="status-wrap">
            <div class="status-grid">
                <div class="status-card">
                    <span class="dot dot-blue"></span>
                    <span class="status-text">정상 : {normal_count:,}</span>
                </div>
                <div class="status-card {'alert' if latest_alarm else ''}">
                    <span class="dot dot-red"></span>
                    <span class="status-text">불량 : {defect_count:,}</span>
                </div>
            </div>
        """,
        unsafe_allow_html=True,
    )

    # -----------------------------------------------------
    # 목표 생산량 달성률
    # -----------------------------------------------------
    current_product_count = count_completed_products(
        seen_df,
        page_col="PageNo",
        complete_page_no=39,
    )

    target_rate = min(current_product_count / target_qty, 1.0)
    target_percent = target_rate * 100

    st.progress(
        target_rate,
        text=f"목표 생산량 달성률: {target_percent:.1f}%  |  현재 생산량: {current_product_count:,} / 목표 생산량: {target_qty:,}",
    )

    # -----------------------------------------------------
    # 요약 지표
    # -----------------------------------------------------
    # 현재 row의 날짜 기준으로 일별 집계
    # 날짜가 바뀌면 current_day_df가 새 날짜 데이터만 포함하므로
    # 일별 생산량과 일별 불량이 자동으로 0부터 다시 누적됩니다.
    current_day_df = get_current_day_df(seen_df, latest_row)

    daily_prod_count = count_completed_products(
        current_day_df,
        page_col="PageNo",
        complete_page_no=39,
    )
    daily_defect_count = int((current_day_df["pred_label"] == 1).sum()) if len(current_day_df) > 0 else 0
    cumulative_defect_rate = defect_count / len(seen_df) if len(seen_df) > 0 else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("현재 판정", "불량" if latest_alarm else "정상")
    c2.metric("누적 불량률", f"{cumulative_defect_rate * 100:.2f}%")
    c3.metric("일별 생산량", f"{daily_prod_count:,} 개")
    c4.metric("일별 불량", f"{daily_defect_count:,} point")

    # -----------------------------------------------------
    # 그래프
    # -----------------------------------------------------
    feature_groups = [
        selected_features[i:i + 2]
        for i in range(0, len(selected_features), 2)
    ]

    for row_idx, row_features in enumerate(feature_groups):
        row_cols = st.columns(len(row_features))

        for col, feature in zip(row_cols, row_features):
            fig = make_chart(window_df, feature, latest_row, latest_alarm)

            with col:
                st.markdown(f'<div class="chart-card {"alert" if latest_alarm else ""}>', unsafe_allow_html=True)
                st.plotly_chart(
                    fig,
                    use_container_width=True,
                    key=f"smooth_chart_{feature}",
                    config={
                        "displayModeBar": False,
                        "scrollZoom": False,
                        "responsive": True,
                    },
                )
                st.markdown("</div>", unsafe_allow_html=True)

    # -----------------------------------------------------
    # 최근 판별 로그
    # -----------------------------------------------------
    st.markdown("#### 최근 판별 로그")

    cols = [
        "WorkingTime",
        "Length",
        "SetPower",
        "SetFrequency",
        "SetDuty",
        "RealPower",
        "Speed",
        "GateOnTime",
        "realpower_up_needed",
        "realpower_down_needed",
        "pred_label",
    ]
    cols = [c for c in cols if c in seen_df.columns]

    recent_log_df = seen_df.tail(8)[cols].copy()

    if "RealPower" in recent_log_df.columns:
        recent_log_df["RealPower"] = recent_log_df.apply(format_realpower_with_adjustment, axis=1)

    # RealPower 열 안에 표시했으므로 보조 계산 컬럼은 화면에서 제거
    recent_log_df = recent_log_df.drop(
        columns=[
            *LOG_HIDDEN_COLUMNS,
            "realpower_up_needed",
            "realpower_down_needed",
            "realpower_up_needed_text",
            "realpower_down_needed_text",
        ],
        errors="ignore"
    )

    if "pred_label" in recent_log_df.columns:
        recent_log_df["pred_label"] = recent_log_df["pred_label"].map({0: "정상", 1: "불량"})

    st.dataframe(
        recent_log_df.style.apply(highlight_defect_rows, axis=1),
        use_container_width=True,
        hide_index=True,
    )


def render_daily_log_content(current_idx: int):
    """
    일별 로그 탭 내용.
    selectbox 같은 widget은 이 함수가 실행되는 fragment 내부에서 직접 생성합니다.
    """
    if len(test_all) == 0:
        st.error("표시할 데이터가 없습니다.")
        return

    current_idx = max(1, min(current_idx, len(test_all)))
    seen_df = test_all.iloc[:current_idx].copy()

    st.markdown("### 일별 판별 로그")

    daily_all_df = seen_df.copy()
    daily_all_df["_work_date"] = pd.to_datetime(daily_all_df["WorkingTime"]).dt.date
    available_dates = sorted(daily_all_df["_work_date"].dropna().unique())

    if len(available_dates) == 0:
        st.info("아직 표시할 일별 로그가 없습니다.")
        return

    # 기존 선택 날짜가 아직 선택 가능하면 유지하고, 아니면 가장 최근 날짜 선택
    current_selected = st.session_state.get("daily_log_date_fixed", available_dates[-1])
    if current_selected not in available_dates:
        current_selected = available_dates[-1]

    selected_date = st.selectbox(
        "조회 날짜 선택",
        options=available_dates,
        index=available_dates.index(current_selected),
        key="daily_log_date_fixed",
    )

    selected_daily_df = daily_all_df[daily_all_df["_work_date"] == selected_date].copy()
    selected_daily_prod = count_completed_products(
        selected_daily_df,
        page_col="PageNo",
        complete_page_no=39,
    )
    selected_daily_defect = int((selected_daily_df["pred_label"] == 1).sum())

    # 불량률은 row 기준으로 계산
    selected_daily_defect_rate = (
        selected_daily_defect / len(selected_daily_df)
        if len(selected_daily_df) > 0
        else 0
    )

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("조회 날짜", str(selected_date))
    d2.metric("일별 생산량", f"{selected_daily_prod:,} 개")
    d3.metric("일별 불량", f"{selected_daily_defect:,} point")
    d4.metric("일별 불량률", f"{selected_daily_defect_rate * 100:.2f}%")

    daily_cols = [
        "WorkingTime",
        "Length",
        "SetPower",
        "SetFrequency",
        "SetDuty",
        "RealPower",
        "Speed",
        "GateOnTime",
        "realpower_up_needed",
        "realpower_down_needed",
        "pred_label",
    ]
    daily_cols = [c for c in daily_cols if c in selected_daily_df.columns]

    view_daily_df = selected_daily_df[daily_cols].copy()

    if "RealPower" in view_daily_df.columns:
        view_daily_df["RealPower"] = view_daily_df.apply(format_realpower_with_adjustment, axis=1)

    # RealPower 열 안에 표시했으므로 보조 계산 컬럼은 화면에서 제거
    view_daily_df = view_daily_df.drop(
        columns=[
            *LOG_HIDDEN_COLUMNS,
            "realpower_up_needed",
            "realpower_down_needed",
            "realpower_up_needed_text",
            "realpower_down_needed_text",
        ],
        errors="ignore"
    )

    if "pred_label" in view_daily_df.columns:
        view_daily_df["pred_label"] = view_daily_df["pred_label"].map({0: "정상", 1: "불량"})

    st.markdown("#### 일별 판별 로그")
    st.dataframe(
        view_daily_df.style.apply(highlight_defect_rows, axis=1),
        use_container_width=True,
        hide_index=True,
        height=420,
    )

    st.markdown("#### 레시피 조합 (Speed, Length, SetPower, SetFrequency, SetDuty, GateOnTime, RealPower)")
    recipe_count_df = build_recipe_combination_count_df(selected_daily_df)

    if len(recipe_count_df) == 0:
        st.info("표시할 레시피 조합 데이터가 없습니다.")
    else:
        st.markdown(
            make_recipe_combination_count_html(recipe_count_df),
            unsafe_allow_html=True,
        )


def render_defect_model_tab():
    st.markdown("### 불량 판정 모델")
    st.caption(f"현재 실시간 모니터링 적용 모델: {active_model_name}")

    if len(modeling_all_base) == 0:
        st.info("모델 학습에 사용할 데이터가 없습니다.")
        return

    valid_groups = model_group_summary["그룹"].tolist()
    max_group = max(valid_groups) if valid_groups else 0
    defect_group_ids = get_defect_group_ids(model_group_summary)
    auto_predict_range_text = format_group_range_text(defect_group_ids)

    summary_view = model_group_summary.copy()
    summary_view = summary_view.drop(columns=["데이터", "행수"], errors="ignore")
    for col in ["시작시간", "종료시간"]:
        summary_view[col] = pd.to_datetime(summary_view[col], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")

    st.markdown("#### 그룹 목록")
    if max_group:
        st.info(f"학습 그룹 번호 범위: 1 ~ {max_group:,}")
    else:
        st.info("학습 그룹 번호가 없습니다.")
    st.dataframe(summary_view, use_container_width=True, hide_index=True, height=220)

    selected_model_name = st.selectbox(
        "1. 학습할 모델 선택",
        DEFECT_MODEL_OPTIONS,
        index=0,
        key="defect_model_choice",
        format_func=lambda model_name: f"🔴 {model_name}" if model_name == active_model_name else model_name,
    )

    default_train_range = f"1-{min(10, max_group)}" if max_group else ""
    train_range_text = st.text_input(
        "2. 학습시킬 그룹 번호",
        value=st.session_state.get("model_train_range_text", default_train_range),
        key="model_train_range_text",
        help="예: 1-10, 23-25 처럼 입력합니다. PageNo 1~39를 한 그룹으로 묶은 번호입니다.",
    )
    train_group_ids, invalid_train_groups = parse_group_selection(train_range_text, valid_groups)
    train_df_for_model = modeling_all_base[modeling_all_base["model_group"].isin(train_group_ids)].copy()
    train_defect_groups = (
        train_df_for_model.groupby("model_group")["true_label"].sum()
        if len(train_df_for_model) else pd.Series(dtype=float)
    )
    train_defect_groups = sorted([int(idx) for idx, value in train_defect_groups.items() if value > 0])

    if invalid_train_groups:
        st.warning(f"존재하지 않는 그룹 번호: {', '.join(map(str, invalid_train_groups[:20]))}")

    if train_defect_groups:
        st.error(f"{', '.join(map(str, train_defect_groups))}번에는 불량 데이터가 포함되어 있습니다. 해당 그룹은 학습시킬 수 없습니다.")

    c1, c2, c3 = st.columns(3)
    c1.metric("선택 학습 그룹", f"{len(train_group_ids):,}개")
    c2.metric("학습 행 수", f"{len(train_df_for_model):,}행")
    c3.metric("학습 그룹 내 불량", f"{int(train_df_for_model['true_label'].sum()) if len(train_df_for_model) else 0:,}행")

    train_clicked = st.button("3. 학습", type="primary", use_container_width=True)
    if train_clicked:
        if not train_group_ids:
            st.error("학습시킬 그룹 번호를 입력해주세요.")
        elif invalid_train_groups:
            st.error("존재하지 않는 그룹 번호가 있어 학습할 수 없습니다.")
        elif train_defect_groups:
            st.error(f"{', '.join(map(str, train_defect_groups))}번에는 불량 데이터가 포함되어 있습니다.")
        else:
            try:
                fitted_model = fit_defect_model(selected_model_name, train_df_for_model, robust_threshold=threshold)
                fitted_model["train_groups"] = train_group_ids
                fitted_model["train_group_text"] = format_group_range_text(train_group_ids)
                fitted_model["signature"] = f"{selected_model_name}|train={','.join(map(str, train_group_ids))}|ts={time.time():.3f}"
                st.session_state["trained_defect_model"] = fitted_model
                st.success(f"{selected_model_name} 학습 완료")

                auto_overlap_groups = sorted(set(train_group_ids).intersection(defect_group_ids))
                if not defect_group_ids:
                    st.warning("불량이 포함된 그룹이 없어 자동 예측을 진행하지 않았습니다.")
                elif auto_overlap_groups:
                    st.error(f"{', '.join(map(str, auto_overlap_groups))}번 그룹은 학습 데이터와 예측 데이터가 겹쳐 과적합 위험성이 있습니다. 예측할 수 없습니다.")
                else:
                    auto_predict_df = modeling_all_base[modeling_all_base["model_group"].isin(defect_group_ids)].copy()
                    prediction_result = build_model_prediction_result(fitted_model, auto_predict_df, defect_group_ids)
                    st.session_state["last_model_prediction"] = prediction_result
                    append_model_training_history(
                        fitted_model,
                        fitted_model["train_group_text"],
                        prediction_result,
                    )
                    st.success("불량 포함 그룹 전체로 자동 예측을 완료했습니다.")
            except Exception as exc:
                st.error(f"학습 실패: {exc}")

    trained_model = st.session_state.get("trained_defect_model")
    if trained_model is not None:
        st.info(
            f"학습 완료 모델: {trained_model.get('name')} / "
            f"학습 그룹 {trained_model.get('train_group_text', format_group_range_text(trained_model.get('train_groups', [])))}"
        )

    st.text_input(
        "4. 예측할 그룹 번호",
        value=auto_predict_range_text,
        disabled=True,
        help="불량 데이터가 포함된 모든 그룹이 자동으로 예측 대상에 설정됩니다.",
    )
    predict_group_ids = defect_group_ids
    invalid_predict_groups = []
    train_group_set = set(trained_model.get("train_groups", [])) if trained_model is not None else set(train_group_ids)
    overlap_groups = sorted(train_group_set.intersection(predict_group_ids))
    predict_df_for_model = modeling_all_base[modeling_all_base["model_group"].isin(predict_group_ids)].copy()

    if invalid_predict_groups:
        st.warning(f"존재하지 않는 예측 그룹 번호: {', '.join(map(str, invalid_predict_groups[:20]))}")

    if overlap_groups:
        st.error(f"{', '.join(map(str, overlap_groups))}번 그룹은 학습 데이터와 예측 데이터가 겹쳐 과적합 위험성이 있습니다. 예측할 수 없습니다.")

    p1, p2, p3 = st.columns(3)
    p1.metric("자동 예측 그룹", f"{len(predict_group_ids):,}개")
    p2.metric("예측 행 수", f"{len(predict_df_for_model):,}행")
    p3.metric("실제 불량", f"{int(predict_df_for_model['true_label'].sum()) if len(predict_df_for_model) else 0:,}행")

    predict_clicked = st.button("5. 자동 예측 다시 실행 및 혼돈행렬 보기", use_container_width=True)
    if predict_clicked:
        if trained_model is None:
            st.error("먼저 모델을 학습시켜주세요.")
        elif not predict_group_ids:
            st.error("예측할 그룹 번호를 입력해주세요.")
        elif invalid_predict_groups:
            st.error("존재하지 않는 그룹 번호가 있어 예측할 수 없습니다.")
        elif overlap_groups:
            st.error("학습 데이터와 예측 데이터가 겹쳐 과적합 위험성이 있습니다.")
        else:
            try:
                prediction_result = build_model_prediction_result(trained_model, predict_df_for_model, predict_group_ids)
                st.session_state["last_model_prediction"] = prediction_result
                append_model_training_history(
                    trained_model,
                    trained_model.get("train_group_text", format_group_range_text(trained_model.get("train_groups", []))),
                    prediction_result,
                )
                render_prediction_result(prediction_result)
            except Exception as exc:
                st.error(f"예측 실패: {exc}")

    last_prediction = st.session_state.get("last_model_prediction")
    if last_prediction is not None and not predict_clicked:
        st.markdown("#### 최근 예측 결과")
        render_prediction_result(last_prediction)

    prediction_ready = (
        trained_model is not None
        and last_prediction is not None
        and last_prediction.get("model_signature") == trained_model.get("signature")
    )
    apply_disabled = trained_model is None or not prediction_ready
    if trained_model is not None and not prediction_ready:
        st.caption("학습된 모델을 적용하려면 먼저 해당 모델로 예측을 진행하고 혼돈행렬을 확인해주세요.")

    if st.button("6. 적용", disabled=apply_disabled, use_container_width=True):
        st.session_state["active_defect_model"] = trained_model
        st.session_state.idx_light = 0
        st.session_state.prev_idx_light = 0
        st.session_state.last_flash_idx = -1
        st.session_state.last_defect_event_key = None
        st.success("학습된 모델을 실시간 모니터링에 적용했습니다.")
        st.rerun()


def render_model_training_history_tab():
    st.markdown("### 모델 학습 기록")

    history = st.session_state.get("model_training_history", [])
    columns = ["모델명", "학습시킨 그룹범위", "recall", "f1-score", "Accuracy"]

    if not history:
        st.info("아직 학습 및 예측 기록이 없습니다.")
        return

    history_df = pd.DataFrame(history).drop(columns=["_기록키"], errors="ignore")
    for col in ["recall", "f1-score", "Accuracy"]:
        history_df[col] = pd.to_numeric(history_df[col], errors="coerce").round(4)

    st.dataframe(
        history_df.reindex(columns=columns),
        use_container_width=True,
        hide_index=True,
        height=420,
    )


# =========================================================
# 자동 재생 - Smooth Fragment 방식
# =========================================================
@st.fragment(run_every=monitor_run_every)
def smooth_monitor_fragment():
    """
    실시간 모니터링 영역만 재생 상태와 데이터 간격에 맞춰 부분 업데이트합니다.
    외부 placeholder에 쓰지 않으므로 FragmentWidgetsNotAllowedOutsideError를 피합니다.
    """
    if st.session_state.is_running_light:
        now = time.time()

        if st.session_state.idx_light < len(test_all) and now >= st.session_state.next_update_time_light:
            st.session_state.prev_idx_light = st.session_state.idx_light

            # WorkingTime 간격 사용 시에는 기본적으로 1 row씩 진행
            # 옵션을 끄면 기존처럼 batch_size만큼 진행
            step_size = 1 if use_workingtime_interval else batch_size

            st.session_state.idx_light = min(
                len(test_all),
                st.session_state.idx_light + step_size,
            )

            if use_workingtime_interval:
                delay_seconds = get_workingtime_delay_seconds(
                    test_all,
                    st.session_state.idx_light,
                    default_seconds=sleep_sec,
                    cap_threshold_seconds=10.0,
                    capped_seconds=max_interval_seconds,
                )
            else:
                delay_seconds = sleep_sec

            st.session_state.next_update_time_light = now + delay_seconds

        elif st.session_state.idx_light >= len(test_all):
            st.session_state.is_running_light = False

    render_monitor_content(st.session_state.idx_light)
    components.html(
        make_scroll_position_guard_html(st.session_state.idx_light, st.session_state.is_running_light),
        height=0,
        scrolling=False,
    )

    if not st.session_state.is_running_light and st.session_state.idx_light >= len(test_all):
        st.success("재생이 종료되었습니다.")


@st.fragment(run_every=daily_log_run_every)
def smooth_daily_log_fragment():
    """
    일별 로그도 fragment 안에서 직접 그리되, 모니터 차트보다 낮은 빈도로 갱신합니다.
    selectbox가 fragment 밖 container에 쓰이지 않도록 분리했습니다.
    """
    current_idx = max(1, min(st.session_state.idx_light, len(test_all)))
    render_daily_log_content(current_idx)


# =========================================================
# 탭 구성
# =========================================================
tab_monitor, tab_daily_log, tab_defect_model, tab_model_history = st.tabs([
    "실시간 모니터링",
    "일별 로그",
    "불량 판정 모델",
    "모델 학습 기록",
])

with tab_monitor:
    smooth_monitor_fragment()

with tab_daily_log:
    smooth_daily_log_fragment()

with tab_defect_model:
    render_defect_model_tab()

with tab_model_history:
    render_model_training_history_tab()
