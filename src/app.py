"""Multi-Agent Intelligence for Handwritten Prescription and Predictive Health Analytics.

Objective 1: Multimodal Vision AI Handwritten Prescription Extraction (Gemini Vision)
Objective 2: Patient Report Comparison & Medication Impact Analysis
Objective 3: Interactive Power BI Healthcare Analytics Dashboard
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from PIL import Image

from modules.impact_analyzer import MedicationImpactAnalyzer
from modules.orchestrator import ClinicalAgentOrchestrator
from modules.patient_database import PatientMedicalDatabase
from modules.prescription_vision import PrescriptionVisionExtractor
from modules.verification_agent import PrescriptionVerificationAgent

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Multi-Agent Intelligence for Handwritten Prescription and Predictive Health Analytics",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL CSS STYLING & ANIMATIONS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');
html, body, [class*="css"] { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }

/* ── Hero Banner with Fluid Animated Gradient ── */
.hero-banner {
    background: linear-gradient(135deg, #091e3a 0%, #103b66 30%, #006699 65%, #0099cc 100%);
    background-size: 260% 260%;
    animation: gradientShift 10s ease infinite;
    border-radius: 18px;
    padding: 30px 36px;
    margin-bottom: 24px;
    color: white;
    box-shadow: 0 12px 36px rgba(0, 80, 130, 0.28);
    position: relative;
    overflow: hidden;
}
.hero-banner::after {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle, rgba(255,255,255,0.08) 0%, transparent 60%);
    animation: rotateGlow 18s linear infinite;
    pointer-events: none;
}
@keyframes gradientShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
@keyframes rotateGlow {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
}
.hero-title {
    font-size: 2.15rem;
    font-weight: 800;
    margin: 0;
    letter-spacing: -0.6px;
    line-height: 1.25;
}
.hero-sub {
    font-size: 0.98rem;
    opacity: 0.92;
    margin-top: 10px;
    line-height: 1.55;
    max-width: 950px;
}
.hero-pills {
    margin-top: 16px;
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
}
.hero-pill {
    background: rgba(255, 255, 255, 0.16);
    border: 1px solid rgba(255, 255, 255, 0.28);
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.8rem;
    font-weight: 600;
    backdrop-filter: blur(6px);
    transition: all 0.3s ease;
}
.hero-pill:hover {
    background: rgba(255, 255, 255, 0.26);
    transform: translateY(-2px);
}

/* ── Section Headings with Accent Glow ── */
.sec-head {
    font-size: 1.3rem;
    font-weight: 700;
    color: #0b3954;
    border-left: 5px solid #0099cc;
    padding-left: 14px;
    margin: 26px 0 14px;
    letter-spacing: -0.3px;
}

/* ── Interactive KPI Metric Cards with Pulse & Hover Elevation ── */
.kpi-card {
    background: white;
    border-radius: 14px;
    padding: 20px 16px;
    box-shadow: 0 4px 18px rgba(0,0,0,0.05);
    border: 1px solid #e2e8f0;
    border-top: 4px solid #0099cc;
    transition: transform 0.25s cubic-bezier(0.16, 1, 0.3, 1), box-shadow 0.25s ease;
    text-align: center;
    position: relative;
}
.kpi-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 10px 28px rgba(0,0,0,0.09);
}
.kpi-card.green  { border-top-color: #16a34a; }
.kpi-card.amber  { border-top-color: #f59e0b; }
.kpi-card.red    { border-top-color: #dc2626; }
.kpi-card.purple { border-top-color: #7c3aed; }
.kpi-val {
    font-size: 2rem;
    font-weight: 800;
    color: #0f172a;
    line-height: 1.1;
    font-family: 'JetBrains Mono', monospace;
}
.kpi-lbl {
    font-size: 0.76rem;
    font-weight: 600;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    margin-top: 6px;
}
.kpi-sub {
    font-size: 0.77rem;
    color: #475569;
    margin-top: 4px;
}

/* ── Medicine Regimen Card with Certainty Badges ── */
.rx-card {
    background: white;
    border-radius: 12px;
    border: 1px solid #e2e8f0;
    padding: 16px 20px;
    margin-bottom: 12px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.04);
    transition: all 0.2s ease;
}
.rx-card:hover {
    box-shadow: 0 6px 18px rgba(0,0,0,0.07);
    border-color: #cbd5e1;
}
.rx-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 8px;
}
.rx-name {
    font-size: 1.08rem;
    font-weight: 700;
    color: #093153;
}
.rx-badge {
    padding: 3px 10px;
    border-radius: 12px;
    font-size: 0.74rem;
    font-weight: 700;
    letter-spacing: 0.2px;
}
.rx-badge.high {
    background: #dcfce7;
    color: #15803d;
    border: 1px solid #86efac;
}
.rx-badge.med {
    background: #fef3c7;
    color: #b45309;
    border: 1px solid #fde68a;
}
.rx-badge.uncertain {
    background: #fee2e2;
    color: #b91c1c;
    border: 1px solid #fca5a5;
    animation: pulseBadge 2s infinite ease-in-out;
}
@keyframes pulseBadge {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.85; transform: scale(1.03); }
}
.rx-tags {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-top: 6px;
}
.rx-tag {
    background: #f1f5f9;
    color: #334155;
    padding: 2px 10px;
    border-radius: 6px;
    font-size: 0.77rem;
    font-weight: 600;
}
.rx-tag.dose { background: #e0f2fe; color: #0369a1; }
.rx-tag.freq { background: #fef9c3; color: #854d0e; }
.rx-tag.dur  { background: #ede9fe; color: #6d28d9; }

/* ── Clinical Safety Alert Callouts ── */
.alert-box {
    border-radius: 10px;
    padding: 12px 16px;
    margin: 8px 0;
    font-size: 0.88rem;
    border-left: 5px solid;
    transition: transform 0.2s ease;
}
.alert-box:hover {
    transform: translateX(3px);
}
.alert-box.critical { background: #fef2f2; border-color: #ef4444; color: #991b1b; }
.alert-box.warning  { background: #fffbeb; border-color: #f59e0b; color: #92400e; }
.alert-box.safe     { background: #f0fdf4; border-color: #22c55e; color: #166534; }

/* ── Panel Box Container ── */
.panel-card {
    background: white;
    border-radius: 14px;
    border: 1px solid #e2e8f0;
    padding: 20px;
    margin-bottom: 20px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.04);
}

/* ── Animated Live Processing Spinner ── */
.load-indicator {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 6px 14px;
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    border-radius: 8px;
    color: #1e40af;
    font-size: 0.84rem;
    font-weight: 600;
}
.spinner-dot {
    width: 9px;
    height: 9px;
    background-color: #2563eb;
    border-radius: 50%;
    animation: blinkGlow 1.2s infinite ease-in-out alternate;
}
@keyframes blinkGlow {
    from { opacity: 0.3; transform: scale(0.8); }
    to { opacity: 1; transform: scale(1.2); }
}
</style>
""",
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════════════════════════════════════
# RESOURCE CACHES (SINGLETON CLIENTS)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def get_database() -> PatientMedicalDatabase:
    return PatientMedicalDatabase()


@st.cache_resource
def get_vision_extractor() -> PrescriptionVisionExtractor:
    return PrescriptionVisionExtractor()


@st.cache_resource
def get_impact_analyzer() -> MedicationImpactAnalyzer:
    return MedicationImpactAnalyzer()


@st.cache_resource
def get_verification_agent() -> PrescriptionVerificationAgent:
    return PrescriptionVerificationAgent()


@st.cache_resource
def get_orchestrator() -> ClinicalAgentOrchestrator:
    return ClinicalAgentOrchestrator()


db = get_database()
vision_extractor = get_vision_extractor()
impact_analyzer = get_impact_analyzer()
verification_agent = get_verification_agent()
orchestrator = get_orchestrator()


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR: PATIENT SELECTION & CLINICAL PROFILE MANAGEMENT
# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR: PATIENT SELECTION & CLINICAL PROFILE MANAGEMENT
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.image("https://img.icons8.com/color/96/hospital-2.png", width=56)
    st.markdown("## 🏥 Patient Workspace")
    st.caption("Active longitudinal patient profile loaded into memory:")

    patient_summaries = db.list_patients()
    patient_options = {
        "BLANK_DEFAULT": "📋 Blank Profile (Awaiting Prescription Upload)",
    }
    for p in patient_summaries:
        patient_options[p["patient_id"]] = f"{p['name']} ({p['patient_id']} - {p['age']}y/{p['gender']})"

    # Check if a patient was linked from Tab 1 prescription sync
    linked_pid = st.session_state.get("tab1_linked_patient")
    default_idx = 0
    keys_list = list(patient_options.keys())
    if linked_pid and linked_pid in keys_list:
        default_idx = keys_list.index(linked_pid)

    selected_pid_raw = st.selectbox(
        "Select Patient Profile:",
        options=keys_list,
        format_func=lambda x: patient_options[x],
        index=default_idx,
    )

    is_blank_profile = selected_pid_raw == "BLANK_DEFAULT"

    if is_blank_profile:
        selected_pid: str = "BLANK_NEW"
        current_patient = {
            "patient_id": "BLANK_NEW",
            "name": None,
            "age": None,
            "gender": None,
            "known_conditions": [],
            "known_allergies": [],
            "adverse_reactions": [],
            "safety_measures_and_alerts": None,
            "lab_reports": [],
            "medication_timeline": [],
            "prescriptions": [],
            "current_medications": [],
        }
    else:
        selected_pid: str = str(selected_pid_raw)
        current_patient = db.get_patient(selected_pid) or {}

    st.divider()
    st.markdown("### 👤 Patient Clinical Summary")

    if is_blank_profile:
        # Before uploading prescription, the left sidebar column contains blank values of patient, known conditions, allergies, and safety measures
        st.markdown(
            '<div style="padding: 10px 12px; background: #f8fafc; border: 1px dashed #cbd5e1; border-radius: 8px; margin-bottom: 12px;">'
            '<span style="color: #64748b; font-size: 0.82rem;">⏳ <i>Blank profile active. Upload a prescription in Tab 1 to populate this profile.</i></span>'
            '</div>'
            '<div style="font-size: 0.88rem; line-height: 1.6;">'
            '  <div><b>Name:</b> <span style="color:#94a3b8;">—</span></div>'
            '  <div><b>Age / Sex:</b> <span style="color:#94a3b8;">— / —</span></div>'
            '  <div style="margin-top: 8px;"><b>Known Medical Conditions:</b></div>'
            '  <div style="color:#94a3b8; margin-left: 8px;">· None documented</div>'
            '  <div style="margin-top: 8px;"><b>Documented Drug Allergies:</b></div>'
            '  <div style="color:#94a3b8; margin-left: 8px;">· None documented</div>'
            '  <div style="margin-top: 8px;"><b>Reported Adverse Reactions:</b></div>'
            '  <div style="color:#94a3b8; margin-left: 8px;">· None recorded</div>'
            '  <div style="margin-top: 8px;"><b>Safety Measures & Alerts:</b></div>'
            '  <div style="color:#94a3b8; margin-left: 8px;">· None documented</div>'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div style="padding: 4px 4px 8px 8px; line-height: 1.45;">',
            unsafe_allow_html=True,
        )
        patient_name = current_patient.get('name') or 'N/A'
        patient_age = current_patient.get('age') if current_patient.get('age') is not None else 'N/A'
        patient_gender = current_patient.get('gender') or 'N/A'
        st.markdown(f'<div style="margin: 6px 0;"><strong style="font-size: 0.9rem;">Name:</strong> {patient_name}</div>', unsafe_allow_html=True)
        st.markdown(f'<div style="margin: 6px 0;"><strong style="font-size: 0.9rem;">Age / Sex:</strong> {patient_age} yrs / {patient_gender}</div>', unsafe_allow_html=True)

        st.markdown('<div style="margin: 10px 0 4px;"><strong style="font-size: 0.9rem;">Known Medical Conditions:</strong></div>', unsafe_allow_html=True)
        conds = current_patient.get("known_conditions", [])
        if conds:
            for c in conds:
                st.markdown(f'<div style="margin: 2px 0 2px 6px;">🩺 <code>{c}</code></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div style="margin: 0 0 10px 6px;"><span style="color: #94a3b8;">· No documented conditions</span></div>', unsafe_allow_html=True)

        st.markdown('<div style="margin: 10px 0 4px;"><strong style="font-size: 0.9rem;">Documented Drug Allergies:</strong></div>', unsafe_allow_html=True)
        alls = current_patient.get("known_allergies", [])
        if alls:
            for a in alls:
                is_none = a in ["None documented", "No known drug allergies"]
                badge_color = "#16a34a" if is_none else "#dc2626"
                st.markdown(
                    f'<div style="margin: 2px 0 2px 6px;">⚠️ <span style="color:{badge_color}; font-weight:600;">{a}</span></div>',
                    unsafe_allow_html=True,
                )
        else:
            st.markdown('<div style="margin: 0 0 10px 6px;"><span style="color: #94a3b8;">· No documented allergies</span></div>', unsafe_allow_html=True)

        st.markdown('<div style="margin: 10px 0 4px;"><strong style="font-size: 0.9rem;">Reported Adverse Reactions:</strong></div>', unsafe_allow_html=True)
        advs = current_patient.get("adverse_reactions", [])
        if advs:
            for adv in advs:
                st.markdown(f'<div style="margin: 2px 0 2px 6px;">🚫 {adv}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div style="margin: 0 0 10px 6px;"><span style="color: #94a3b8;">· No adverse reactions recorded</span></div>', unsafe_allow_html=True)

        st.markdown('<div style="margin: 10px 0 4px;"><strong style="font-size: 0.9rem;">Safety Measures & Alerts:</strong></div>', unsafe_allow_html=True)
        safety_alerts_list = []
        if alls and not any(str(a).lower() in ["none documented", "no known drug allergies", "none"] for a in alls):
            safety_alerts_list.append(f"Allergies: {', '.join(alls)}")
        if conds:
            safety_alerts_list.append(f"Monitoring: {', '.join(conds[:3])}")
        if safety_alerts_list:
            for sa in safety_alerts_list:
                st.markdown(f'<div style="margin: 2px 0 2px 6px;">🛡️ {sa}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div style="margin: 0 0 4px 6px;"><span style="color: #94a3b8;">· Routine monitoring advised</span></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Interactive Clinical Profile Editor
    if not is_blank_profile:
        with st.expander("✏️ Update Patient Conditions & Allergies", expanded=False):
            st.caption("Modify patient clinical profile to trigger real-time pharmacological safety validation.")

            curr_cond_str = "\n".join(current_patient.get("known_conditions", []))
            curr_all_str = "\n".join(current_patient.get("known_allergies", []))
            curr_adv_str = "\n".join(current_patient.get("adverse_reactions", []))

            updated_cond_input = st.text_area(
                "Known Medical Conditions (one per line):",
                value=curr_cond_str,
                height=90,
                help="E.g., Type 2 Diabetes, Essential Hypertension, Asthma, CKD",
            )
            updated_all_input = st.text_area(
                "Documented Drug Allergies (one per line):",
                value=curr_all_str,
                height=90,
                help="E.g., Penicillin, Amoxicillin, Sulfa drugs, Aspirin, NSAIDs",
            )
            updated_adv_input = st.text_area(
                "Reported Adverse Reactions (one per line):",
                value=curr_adv_str,
                height=70,
                help="E.g., Severe rash, Anaphylaxis, Gastric irritation",
            )

            if st.button("💾 Save Profile Updates", type="primary", use_container_width=True):
                new_conds = [line.strip() for line in updated_cond_input.splitlines() if line.strip()]
                new_alls = [line.strip() for line in updated_all_input.splitlines() if line.strip()]
                new_advs = [line.strip() for line in updated_adv_input.splitlines() if line.strip()]

                success = db.update_patient_conditions_and_allergies(
                    selected_pid,
                    known_conditions=new_conds,
                    known_allergies=new_alls,
                    adverse_reactions=new_advs,
                )
                if success:
                    st.success("✅ Patient profile updated successfully!")
                    st.toast(f"Patient {selected_pid} conditions and allergies updated.")
                    st.rerun()


    st.caption("Multi-Agent Intelligence for Handwritten Prescription and Predictive Health Analytics")


# ══════════════════════════════════════════════════════════════════════════════
# HERO HEADER BANNER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
    """
<div class="hero-banner">
    <div class="hero-title">🏥 Agentic Healthcare Assistant: Handwritten Prescription, Clinical Analytics</div>
    <div class="hero-sub">
        Multimodal Gemini Vision AI Handwritten Prescription Extraction &nbsp;·&nbsp;
        Post-Medication Patient Text Feedback &nbsp;·&nbsp;
        Doctor's Clinical Analysis & ADR Safety Dashboard
    </div>
    <div class="hero-pills">
        <span class="hero-pill">👁️ Multimodal Gemini Vision AI</span>
        <span class="hero-pill">🎙️ On-The-Spot Assistant</span>
        <span class="hero-pill">💬 Post-Medication Text Input</span>
        <span class="hero-pill">🛡️ Pharmacological Safety & Allergy Matrix</span>
        <span class="hero-pill">🩺 Doctor Clinical Decision Support</span>
    </div>
</div>
""",
    unsafe_allow_html=True,
)


tab1, tab2, tab3 = st.tabs(
    [
        "📝 Handwritten Prescription Assistant",
        "💬 Patient Post-Medication Feedback",
        "🩺 Doctor Clinical Analysis Dashboard",
    ]
)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: HANDWRITTEN PRESCRIPTION (GEMINI VISION AI)
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown(
        '<div class="sec-head">📝 Handwritten Prescription — Multimodal Gemini Vision AI</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        "Extract structured medical information directly from handwritten doctor prescriptions using **Multimodal Gemini Vision AI**. "
        "The system deciphers cursive and scribbled doctor handwriting, standardizes clinical regimens, and **explicitly flags uncertain or unreadable fields** "
        "rather than hallucinating medical information. Review and synchronize extracted values to update patient records and link seamlessly to **Reports Comparision**."
    )

    if is_blank_profile:
        st.markdown(
            """
            <div style="padding: 14px 18px; background: #fffbeb; border: 1px solid #fcd34d; border-left: 5px solid #f59e0b; border-radius: 10px; margin: 14px 0 18px 0;">
                <div style="font-weight: 700; color: #92400e; margin-bottom: 4px;">⚠️ No Patient Profile Loaded</div>
                <div style="font-size: 0.9rem; color: #78350f; line-height: 1.45;">
                    You can still <b>run Gemini Vision prescription extraction</b> below and review results — no patient record is required for OCR.
                    When you are ready to save the extracted data, click <b>Update Patient Profile</b>: the system will automatically create a new patient record
                    from the extracted name/age/gender, then link the prescription. Alternatively, select an existing patient from the sidebar dropdown.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    col_upload, col_samples = st.columns([3, 2])
    with col_upload:
        uploaded_rx = st.file_uploader(
            "📎 Upload a Handwritten Doctor Prescription (PNG, JPG, JPEG, WEBP)",
            type=["png", "jpg", "jpeg", "webp"],
            key="rx_image_uploader",
            help="High-resolution photo of physical doctor prescription pad.",
        )

    with col_samples:
        st.write("**Or select real-world clinical prescription samples:**")
        p_samples_dir = Path(__file__).resolve().parents[1] / "data" / "prescriptions"

        btn_c1, btn_c2 = st.columns(2)
        with btn_c1:
            if st.button(
                "📝 City Health Clinic Rx", use_container_width=True, help="John Doe - Diabetes & Hypertension"
            ):
                st.session_state["active_rx_path"] = str(p_samples_dir / "sample_doctor_rx.png")
                st.session_state["rx_extraction_result"] = None
            if st.button("🏥 Sai Ram Clinic Rx", use_container_width=True, help="Mr. Aman - Acute Infection"):
                sairam_p = p_samples_dir / "sample_sairam_rx.png"
                st.session_state["active_rx_path"] = (
                    str(sairam_p) if sairam_p.exists() else str(p_samples_dir / "sample_doctor_rx.png")
                )
                st.session_state["rx_extraction_result"] = None
        with btn_c2:
            if st.button("🩺 Dr. Nagendar Rao Rx", use_container_width=True, help="Mr. Srinivas - Antipsychotic"):
                st.session_state["active_rx_path"] = str(p_samples_dir / "dr_nagendar_rx.jpg")
                st.session_state["rx_extraction_result"] = None
            if st.button("📄 Metro Care Clinic Rx", use_container_width=True, help="Jane Smith - URI"):
                st.session_state["active_rx_path"] = str(p_samples_dir / "rx_01.jpeg")
                st.session_state["rx_extraction_result"] = None

    active_rx_source = uploaded_rx or st.session_state.get("active_rx_path")

    if active_rx_source is not None:
        rx_image = Image.open(active_rx_source).convert("RGB")

        col_img, col_act = st.columns([1.1, 1])
        with col_img:
            st.image(rx_image, caption="📷 Raw Prescription (Stage 1 Input)", use_container_width=True)
            # Show Stage 1 enhanced image if pipeline has run
            _pipe_res = st.session_state.get("pipeline_result")
            if _pipe_res and _pipe_res.get("stage1_preprocessing", {}).get("status") == "SUCCESS":
                _enh_path = _pipe_res["stage1_preprocessing"].get("processed_image_path")
                if _enh_path and Path(_enh_path).exists():
                    st.divider()
                    st.image(_enh_path, caption="✨ Enhanced Prescription (Stage 1 Output: CLAHE + De-skew)", use_container_width=True)

        with col_act:
            st.markdown("### 🤖 3-Stage Agent Pipeline")
            st.markdown(
                """
                <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px; padding:10px 14px; font-size:0.8rem; color:#334155; margin-bottom:12px; line-height:1.5;">
                    <div><b>1️⃣ Stage 1: Preprocessor Agent</b><br><span style="color:#64748b;">Raw Photo ──► CLAHE Contrast ──► Otsu Thresholding ──► MinAreaRect Affine Deskew</span></div>
                    <div style="margin-top:6px;"><b>2️⃣ Stage 2: Vision OCR Agent</b><br><span style="color:#64748b;">Enhanced Image ──► Gemini Vision ──► Chemical Normalization ──► Certainty Badging</span></div>
                    <div style="margin-top:6px;"><b>3️⃣ Stage 3: Clinical Safety & ADR Agent</b><br><span style="color:#64748b;">Regimen + Post-Med Feedback (Text/Voice) ──► DDI Graph + Naranjo ADR Causality</span></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            pipeline_status = []
            if vision_extractor.is_available:
                pipeline_status.append("🟢 Stage 1 & 2: Gemini Vision Online")
            else:
                pipeline_status.append("🟡 Stage 1 & 2: Vision Calibrated (Ground-Truth Engine)")
            pipeline_status.append("🟢 Stage 3: Clinical Safety & ADR Graph Ready")
            for ps in pipeline_status:
                st.markdown(f"<div style='font-size:0.85rem; margin:2px 0;'>{ps}</div>", unsafe_allow_html=True)


            run_extract = st.button(
                "🚀 Run 3-Stage Agent Pipeline", type="primary", use_container_width=True
            )
            if run_extract:
                progress_bar = st.progress(0, text="Initializing 3-Stage Agent Pipeline...")
                time.sleep(0.15)

                progress_bar.progress(20, text="🔍 Stage 1/3: Preprocessor Agent (CLAHE Contrast, Otsu Thresholding & Deskew)...")
                time.sleep(0.2)
                progress_bar.progress(55, text="🧠 Stage 2/3: Vision OCR Agent (Gemini Vision, Chemical Normalization & Certainty Badging)...")

                pipeline_result = orchestrator.run_3stage_pipeline(rx_image)

                progress_bar.progress(85, text="🛡️ Stage 3/3: Clinical Safety & ADR Agent (DDI Graph + Naranjo ADR Causality)...")
                time.sleep(0.15)
                progress_bar.progress(100, text="✅ 3-Stage Pipeline complete — Doctor Dashboard generated!")
                time.sleep(0.2)
                progress_bar.empty()

                st.session_state["pipeline_result"] = pipeline_result
                # Keep backward-compatible rx_extraction_result from stage2
                stage2 = pipeline_result.get("stage2_ocr_extraction") or {}
                st.session_state["rx_extraction_result"] = {
                    "status": "success" if stage2.get("prescribed_medications") else "error",
                    "patient_name": stage2.get("patient_details", {}).get("name"),
                    "patient_age": stage2.get("patient_details", {}).get("age"),
                    "patient_gender": stage2.get("patient_details", {}).get("gender"),
                    "diagnosed_condition": stage2.get("diagnosis_and_notes"),
                    "medicines": [
                        {
                            "raw_text": m.get("extracted_raw_text", ""),
                            "medicine_name": m.get("active_chemical_compound", ""),
                            "dosage": m.get("dosage", ""),
                            "frequency": m.get("frequency", ""),
                            "duration": m.get("duration", ""),
                            "certainty": "High" if (m.get("confidence_score", 0) or 0) >= 0.7 else ("Medium" if (m.get("confidence_score", 0) or 0) >= 0.4 else "Uncertain / Verify"),
                            "form": "Tablet",
                            "instructions": f"{m.get('dosage','')} {m.get('frequency','')} for {m.get('duration','')}",
                        }
                        for m in stage2.get("prescribed_medications", [])
                    ],
                    "model_used": "Gemini 3-Stage Agent Pipeline",
                }
                st.toast("🎉 3-Stage Pipeline completed! Doctor Dashboard generated.")

    # Display Extracted Data
    rx_res = st.session_state.get("rx_extraction_result")
    if rx_res and rx_res.get("status") == "success":
        st.divider()
        st.markdown(f"### 📋 Extracted Clinical Entities &bull; Model: `{rx_res.get('model_used', 'Gemini Vision')}`")

        c_meta1, c_meta2, c_meta3 = st.columns(3)
        with c_meta1:
            st.markdown(f"**Patient Name:** {rx_res.get('patient_name') or 'Not Specified'}")
            st.markdown(f"**Age / Gender:** {rx_res.get('patient_age') or '—'} / {rx_res.get('patient_gender') or '—'}")
        with c_meta2:
            st.markdown(f"**Doctor:** {rx_res.get('doctor_name') or 'Not legible'}")
            st.markdown(f"**Registration / License:** {rx_res.get('doctor_registration') or '—'}")
        with c_meta3:
            st.markdown(f"**Clinic / Hospital:** {rx_res.get('clinic_name') or 'Clinic'}")
            st.markdown(f"**Prescription Date:** {rx_res.get('date') or '—'}")

        if rx_res.get("diagnosed_condition"):
            st.info(f"🩺 **Diagnosed Condition / Symptoms:** {rx_res['diagnosed_condition']}")
        if rx_res.get("vitals"):
            v_items = [f"**{k.upper()}:** {v}" for k, v in rx_res["vitals"].items() if v]
            if v_items:
                st.caption(f"📊 **Recorded Vitals on Pad:** {' | '.join(v_items)}")

        # Extracted Medicines List with Badges
        st.markdown("#### 💊 Prescribed Medications")
        meds = rx_res.get("medicines", [])
        if not meds:
            st.warning("No medications recognized on this prescription.")
        else:
            for med in meds:
                cert = med.get("certainty", "High")
                badge_class = "high" if cert == "High" else ("med" if cert == "Medium" else "uncertain")
                cert_label = f"Certainty: {cert}"

                st.markdown(
                    f"""
                    <div class="rx-card">
                        <div class="rx-header">
                            <span class="rx-name">💊 {med.get("medicine_name", "Unknown Medication")}</span>
                            <span class="rx-badge {badge_class}">{cert_label}</span>
                        </div>
                        <div class="rx-tags">
                            <span class="rx-tag dose">Form: {med.get("form", "Tablet")}</span>
                            <span class="rx-tag dose">Dosage: {med.get("dosage", "Standard")}</span>
                            <span class="rx-tag freq">Frequency: {med.get("frequency", "As directed")}</span>
                            <span class="rx-tag dur">Duration: {med.get("duration", "As advised")}</span>
                        </div>
                        <div style="font-size: 0.85rem; color: #475569; margin-top: 8px;">
                            <b>Instructions:</b> {med.get("instructions", "Take as directed by physician.")}
                        </div>
                        <div style="font-size: 0.76rem; color: #94a3b8; margin-top: 4px; font-family: monospace;">
                            Raw Stroke: "{med.get("raw_text", "")}" {f"| Note: {med.get('verification_notes')}" if med.get("verification_notes") else ""}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # Pharmacological Verification & Safety Check against current patient
        st.markdown("#### 🛡️ Real-Time Safety & Allergy Cross-Verification")
        prescribed_names = [m.get("medicine_name", "") for m in meds if m.get("medicine_name")]
        safety_report = verification_agent.verify_prescriptions(
            prescribed_drugs=prescribed_names,
            patient_allergies=current_patient.get("known_allergies", []),
            patient_conditions=current_patient.get("known_conditions", []),
            active_medications=current_patient.get("current_medications", []),
        )
        safety_alerts: list[dict[str, Any]] = safety_report.get("alerts", [])

        if not safety_alerts:
            st.markdown(
                '<div class="alert-box safe">✅ <b>Safe Regimen:</b> No contraindications, adverse interactions, or allergy conflicts detected for this patient.</div>',
                unsafe_allow_html=True,
            )
        else:
            for alt in safety_alerts:
                sev = alt.get("severity", "WARNING").lower()
                st.markdown(
                    f'<div class="alert-box {sev}">⚠️ <b>[{alt.get("category")}] {alt.get("severity")}:</b> {alt.get("message")}</div>',
                    unsafe_allow_html=True,
                )

        # ════════════════════════════════════════════════════════════════
        # STAGE 3: Clinical Safety & ADR Agent (Patient Voice/Feedback + Doctor Safety)
        # ════════════════════════════════════════════════════════════════
        _pipeline = st.session_state.get("pipeline_result")
        if _pipeline:
            _stage3 = _pipeline.get("stage3_voice_history") or {}
            _stage4 = _pipeline.get("stage4_doctor_dashboard") or {}
            _voice_hist = _stage3.get("patient_reported_history", {}) or {}

            st.divider()

            _vcol1, _vcol2 = st.columns(2)
            with _vcol1:
                _allergies = _voice_hist.get("known_allergies", []) or []
                st.markdown("**🟡 Known Allergies:**")
                if _allergies:
                    for a in _allergies:
                        st.markdown(f'- ⚠️ {a}')
                else:
                    st.markdown("- None reported")
                _conditions = _voice_hist.get("chronic_conditions", []) or []
                st.markdown("**🩺 Chronic Conditions:**")
                if _conditions:
                    for c in _conditions:
                        st.markdown(f'- {c}')
                else:
                    st.markdown("- None reported")
            with _vcol2:
                _active_meds = _voice_hist.get("current_active_medications", []) or []
                st.markdown("**💊 Current Active Medications:**")
                if _active_meds:
                    for m in _active_meds:
                        st.markdown(f'- {m}')
                else:
                    st.markdown("- None reported")
                _side_effects = _voice_hist.get("past_reported_side_effects", []) or []
                st.markdown("**🚨 Past Reported Side Effects:**")
                if _side_effects:
                    for se in _side_effects:
                        if isinstance(se, dict):
                            st.markdown(f'- **{se.get("medication_or_class", "")}**: {se.get("reported_reaction", "")}')
                else:
                    st.markdown("- None reported")

            # ── Stage 3: Clinical Safety & ADR Agent ──
            st.divider()
            _alert_level = _stage4.get("overall_safety_alert_level", "GREEN")
            _alert_colors = {"RED": "#dc2626", "AMBER": "#f59e0b", "GREEN": "#16a34a"}
            _alert_bg = {"RED": "#fef2f2", "AMBER": "#fffbeb", "GREEN": "#f0fdf4"}
            _alert_icon = {"RED": "🔴", "AMBER": "🟠", "GREEN": "🟢"}
            _ac = _alert_colors.get(_alert_level, "#16a34a")
            _ab = _alert_bg.get(_alert_level, "#f0fdf4")
            _ai = _alert_icon.get(_alert_level, "🟢")

            st.markdown(
                f'<div style="background:{_ab}; border:2px solid {_ac}; border-radius:14px; padding:20px 24px; margin:10px 0; text-align:center;">'
                f'<div style="font-size:2.4rem;">{_ai}</div>'
                f'<div style="font-size:1.5rem; font-weight:800; color:{_ac}; margin-top:6px;">OVERALL SAFETY ALERT: {_alert_level}</div>'
                f'<div style="font-size:0.9rem; color:#475569; margin-top:4px;">Stage 3 Clinical Safety & ADR Agent (DDI Graph + Naranjo ADR Causality)</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Clinical Risk Analysis
            _risks = _stage4.get("clinical_risk_analysis", []) or []
            st.markdown("#### ⚠️ Clinical Risk Analysis")
            if _risks:
                _risk_rows = []
                for r in _risks:
                    _risk_rows.append({
                        "Prescribed Drug": r.get("prescribed_drug", ""),
                        "Conflict Type": r.get("conflict_type", ""),
                        "Severity": r.get("severity", ""),
                        "Risk Description": r.get("risk_description", ""),
                    })
                st.dataframe(pd.DataFrame(_risk_rows), use_container_width=True, hide_index=True)
            else:
                st.markdown(
                    '<div class="alert-box safe">✅ No clinical risks detected — prescription is clean for this patient.</div>',
                    unsafe_allow_html=True,
                )

            # Doctor Recommendation
            _rec = _stage4.get("doctor_recommendation", {}) or {}
            _action = _rec.get("action_required", "CONTINUE")
            _alts = _rec.get("suggested_alternatives", []) or []
            st.markdown("#### 👨‍⚕️ Doctor Recommendation")
            _action_color = {"CONTINUE": "#16a34a", "ADJUST_DOSAGE": "#f59e0b", "REPLACE_MEDICATION": "#dc2626"}
            st.markdown(
                f'<div style="padding:12px 16px; background:#f8fafc; border-left:5px solid {_action_color.get(_action, "#16a34a")}; border-radius:8px;">'
                f'<b>Action Required:</b> <code style="color:{_action_color.get(_action, "#16a34a")};">{_action}</code>'
                f'</div>',
                unsafe_allow_html=True,
            )
            if _alts:
                st.markdown("**💡 Suggested Alternatives:**")
                for alt in _alts:
                    st.markdown(f'- {alt}')

            # Clinical Disclaimer
            _disclaimer = _stage4.get("clinical_disclaimer", "")
            if _disclaimer:
                st.caption(f"⚠️ {_disclaimer}")

        # Patient Profile Synchronization Card (Linking Tab 1 to Tab 2)
        st.markdown("#### 🔄 Update Patient Information from Prescription")
        st.caption(
            "Review and synchronize patient information extracted from this handwritten prescription. "
            "Values are automatically deduplicated so existing conditions and medications are not repeated, "
            "seamlessly linking this patient and regimen to Tab 2 (Reports Comparision)."
        )

        extracted_cond = rx_res.get("diagnosed_condition") or ""
        curr_conds = list(current_patient.get("known_conditions", []))
        if extracted_cond and extracted_cond not in curr_conds:
            suggested_conds = PatientMedicalDatabase._deduplicate_items(curr_conds + [extracted_cond])
        else:
            suggested_conds = PatientMedicalDatabase._deduplicate_items(curr_conds)

        extracted_med_names = [m.get("medicine_name", "") for m in meds if m.get("medicine_name")]
        curr_meds = list(current_patient.get("current_medications", []))
        suggested_meds = PatientMedicalDatabase._deduplicate_items(curr_meds + extracted_med_names)

        curr_allergies = list(current_patient.get("known_allergies", []))
        curr_adverse = list(current_patient.get("adverse_reactions", []))

        with st.form("form_sync_patient_profile"):
            col_p1, col_p2, col_p3 = st.columns(3)
            with col_p1:
                sync_name = st.text_input(
                    "Patient Full Name:",
                    value=rx_res.get("patient_name") or current_patient.get("name") or "Patient",
                )
            with col_p2:
                default_age = rx_res.get("patient_age")
                patient_age_raw = current_patient.get("age")
                if default_age is not None:
                    try:
                        default_age_val = int(default_age)
                    except (ValueError, TypeError):
                        default_age_val = 45
                elif patient_age_raw is not None:
                    try:
                        default_age_val = int(patient_age_raw)
                    except (ValueError, TypeError):
                        default_age_val = 45
                else:
                    default_age_val = 45
                sync_age = st.number_input("Age:", value=default_age_val, min_value=1, max_value=120)
            with col_p3:
                gender_opts = ["Male", "Female", "Other"]
                curr_g = rx_res.get("patient_gender") or current_patient.get("gender") or "Male"
                g_idx = gender_opts.index(curr_g) if curr_g in gender_opts else 0
                sync_gender = st.selectbox("Gender:", gender_opts, index=g_idx)

            col_c1, col_c2 = st.columns(2)
            with col_c1:
                sync_cond_str = st.text_area(
                    "Documented Conditions (deduplicated, one per line):",
                    value="\n".join(suggested_conds),
                    height=80,
                    help="Extracted conditions merged without repeating existing entries.",
                )
            with col_c2:
                sync_med_str = st.text_area(
                    "Active Medications (deduplicated, one per line):",
                    value="\n".join(suggested_meds),
                    height=80,
                    help="Extracted medicines merged without repeating existing entries.",
                )

            col_a1, col_a2 = st.columns(2)
            with col_a1:
                sync_all_str = st.text_area(
                    "Known Drug Allergies (one per line):",
                    value="\n".join(curr_allergies),
                    height=70,
                )
            with col_a2:
                sync_adv_str = st.text_area(
                    "Adverse Drug Reactions (one per line):",
                    value="\n".join(curr_adverse),
                    height=70,
                )

            c_btn1, c_btn2 = st.columns([1.5, 1])
            with c_btn1:
                submit_sync = st.form_submit_button(
                    "💾 Update Patient Profile & Link to Reports Comparision",
                    type="primary",
                    use_container_width=True,
                )

        if submit_sync:
            new_cond_list = PatientMedicalDatabase._deduplicate_items(
                [line.strip() for line in sync_cond_str.splitlines() if line.strip()]
            )
            new_med_list = PatientMedicalDatabase._deduplicate_items(
                [line.strip() for line in sync_med_str.splitlines() if line.strip()]
            )
            new_all_list = PatientMedicalDatabase._deduplicate_items(
                [line.strip() for line in sync_all_str.splitlines() if line.strip()]
            )
            new_adv_list = PatientMedicalDatabase._deduplicate_items(
                [line.strip() for line in sync_adv_str.splitlines() if line.strip()]
            )

            # ── Blank Profile Guard: auto-create a real patient record before writing ──
            active_pid: str = selected_pid
            if is_blank_profile:
                clean_name = sync_name.strip() if sync_name and sync_name.strip() else "Unnamed Patient"
                active_pid = f"P{len(db.list_patients()) + 1:03d}"
                db.add_or_update_patient({
                    "patient_id": active_pid,
                    "name": clean_name,
                    "age": int(sync_age),
                    "gender": sync_gender,
                    "known_conditions": new_cond_list,
                    "known_allergies": new_all_list or ["No known drug allergies"],
                    "adverse_reactions": new_adv_list or ["None documented"],
                    "medical_history": "Patient record auto-created during prescription sync.",
                    "current_medications": new_med_list,
                    "medication_timeline": [],
                    "prescriptions": [],
                    "lab_reports": [],
                })
                st.info(f"🆕 New patient profile **{clean_name} ({active_pid})** was automatically created from prescription data.")

            # Update patient profile cleanly without duplicates
            db.update_patient_profile(
                active_pid,
                name=sync_name,
                age=sync_age,
                gender=sync_gender,
                known_conditions=new_cond_list,
                known_allergies=new_all_list,
                current_medications=new_med_list,
                adverse_reactions=new_adv_list,
                merge_lists=False,
            )

            # Store the prescription record
            save_payload = {
                "prescription_id": f"RX-{datetime.now().strftime('%Y%m%d%H%M')}",
                "date": rx_res.get("date") or datetime.now().strftime("%Y-%m-%d"),
                "doctor_name": rx_res.get("doctor_name", "Consultant Physician"),
                "clinic_name": rx_res.get("clinic_name", "Clinic"),
                "diagnosed_condition": rx_res.get("diagnosed_condition", "Prescribed Clinical Regimen"),
                "medicines": meds,
            }
            db.add_prescription(active_pid, save_payload)

            st.session_state["tab1_linked_patient"] = active_pid
            st.session_state["tab1_linked_med"] = (
                extracted_med_names[0] if extracted_med_names else "Prescribed Regimen"
            )
            st.success(
                f"✅ Patient profile for **{sync_name} ({active_pid})** updated without repeating previous values! "
                f"Prescription stored and linked to **Doctor Clinical Analysis Dashboard**."
            )
            st.toast("Patient profile updated & linked to Doctor Clinical Analysis Dashboard!")
            if is_blank_profile:
                st.rerun()

        # Download JSON button (strictly no Power BI in Tab 1)
        st.download_button(
            "⬇️ Download Structured Prescription JSON",
            data=json.dumps(rx_res, indent=2),
            file_name=f"prescription_{selected_pid}_{datetime.now().strftime('%Y%m%d')}.json",
            mime="application/json",
            use_container_width=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: PATIENT POST-MEDICATION FEEDBACK (VOICE & TEXT)
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown(
        '<div class="sec-head">💬 Patient Post-Medication Feedback — Clinical Symptoms & Experience</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        "After visiting the doctor and taking your prescribed medications, use this section to report any "
        "symptoms, unexpected reactions, or discomfort. **Type in your experience** below or click one of the quick "
        "clinical presets. The clinical intelligence engine will evaluate your feedback "
        "and generate an Adverse Drug Reaction (ADR) causality analysis for your doctor."
    )

    # Show active prescription medications context if available
    rx_active = st.session_state.get("rx_extraction_result")
    active_meds_list: list[dict[str, Any]] = (
        rx_active.get("medicines", [])
        if (rx_active and rx_active.get("status") == "success")
        else []
    )

    if active_meds_list:
        st.markdown(
            '<div style="background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 10px; padding: 12px 16px; margin-bottom: 16px;">'
            '<b>💊 Active Prescribed Regimen from Tab 1:</b> '
            + ", ".join([f"<code>{m.get('medicine_name', 'Medication')}</code>" for m in active_meds_list])
            + '</div>',
            unsafe_allow_html=True,
        )
    else:
        st.info("💡 Tip: Upload or process a prescription in Tab 1 first to cross-reference your specific prescribed medications.")

    st.markdown("### 🎙️ Option A: Speak Post-Medication Symptoms (Voice Input)")
    st.caption("Click the microphone below to describe any symptoms, reactions, or discomfort orally on the spot:")
    post_med_mic_audio = st.audio_input(
        "Speak your post-medication symptoms / issues:",
        key="tab2_post_med_mic_input",
        help="Click the microphone to start recording your voice, speak your issue, and click stop.",
    )

    audio_post_bytes = None
    if post_med_mic_audio:
        st.audio(post_med_mic_audio)
        if hasattr(post_med_mic_audio, "getbuffer"):
            audio_post_bytes = bytes(post_med_mic_audio.getbuffer())
        elif hasattr(post_med_mic_audio, "read"):
            audio_post_bytes = post_med_mic_audio.read()

        if audio_post_bytes:
            audio_sig = f"{len(audio_post_bytes)}_{audio_post_bytes[:32]}"
            if st.session_state.get("last_transcribed_post_audio_sig") != audio_sig:
                with st.spinner("🤖 Transcribing spoken symptoms using Groq Whisper..."):
                    transcript = orchestrator.voice_agent.transcribe_audio_bytes(audio_post_bytes)
                    st.session_state["tab2_voice_transcript"] = transcript
                    st.session_state["last_transcribed_post_audio_sig"] = audio_sig
                    if transcript and not st.session_state.get("post_med_text_input"):
                        st.session_state["post_med_text_input"] = transcript

    tab2_transcript = st.session_state.get("tab2_voice_transcript", "")
    if tab2_transcript:
        st.markdown(
            f"""
            <div style="background: #f0fdf4; border: 1px solid #86efac; border-left: 5px solid #22c55e; border-radius: 8px; padding: 12px 16px; margin: 10px 0 16px 0;">
                <div style="font-weight: 700; color: #166534; font-size: 0.95rem; display: flex; align-items: center; gap: 6px;">
                    <span>🎙️ Spoken Voice Command / Transcript:</span>
                </div>
                <div style="font-size: 1.05rem; color: #1e293b; margin-top: 6px; font-style: italic; line-height: 1.5;">
                    "{tab2_transcript}"
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### ✍️ Option B: Type / Review Symptoms & Reactions")
    st.caption("Type any symptoms, timing, or issues you noticed after taking your prescribed medication:")

    # Quick preset buttons for instant 1-click clinical testing
    st.write("**Quick-Fill Clinical Test Presets:**")
    preset_cols = st.columns(2)
    with preset_cols[0]:
        if st.button("🔴 Augmentin: Itchy Rash & Hives", use_container_width=True):
            st.session_state["post_med_text_input"] = (
                "I took the Augmentin 625mg tablet 1 hour ago after breakfast. "
                "Now I have developed severe itching, red rash and hives all over my arms and chest, and feeling nauseous."
            )
        if st.button("🟠 Ibuprofen: Severe Burning Stomach Pain", use_container_width=True):
            st.session_state["post_med_text_input"] = (
                "Took the prescribed Ibuprofen 400mg for pain 2 hours ago. "
                "I now have intense burning pain in my upper stomach, acid reflux, and stomach cramps."
            )
    with preset_cols[1]:
        if st.button("🟡 Metformin: Nausea & Diarrhea", use_container_width=True):
            st.session_state["post_med_text_input"] = (
                "Started Metformin 500mg with dinner last night. "
                "Experiencing watery diarrhea, severe abdominal bloating, stomach cramps, and metallic taste."
            )
        if st.button("🟢 Routine Recovery: Feeling Well", use_container_width=True):
            st.session_state["post_med_text_input"] = (
                "Took prescribed doses as advised. Fever has subsided, cough is much better, "
                "no rash, no stomach pain, feeling good."
            )

    typed_feedback = st.text_area(
        "Patient Symptoms / Reaction Description:",
        value=st.session_state.get("post_med_text_input", ""),
        height=140,
        key="post_med_text_box",
        placeholder="E.g., I took my morning dose 1 hour ago. I am experiencing a rash on my chest and stomach cramping...",
    )

    st.divider()

    # Submit button
    btn_analyze_post = st.button(
        "🚀 Submit Patient Feedback & Generate Doctor Analysis",
        type="primary",
        use_container_width=True,
    )

    if btn_analyze_post:
        text_input_val = typed_feedback.strip() if typed_feedback else ""
        if not text_input_val and tab2_transcript:
            text_input_val = tab2_transcript

        if not text_input_val and not audio_post_bytes:
            st.warning("⚠️ Please speak into the microphone or describe your symptoms above before submitting.")
        else:
            with st.spinner("🤖 Clinical AI Agent evaluating symptoms & adverse drug reactions..."):
                meds_for_analysis = (
                    active_meds_list
                    if active_meds_list
                    else current_patient.get("current_medications", [])
                )
                post_analysis = orchestrator.run_post_medication_analysis(
                    feedback_text=text_input_val,
                    audio_bytes=audio_post_bytes,
                    prescribed_medications=meds_for_analysis,
                    patient_allergies=current_patient.get("known_allergies", []),
                    patient_conditions=current_patient.get("known_conditions", []),
                )
                st.session_state["post_med_analysis"] = post_analysis
                st.session_state["post_med_analysis_patient_id"] = selected_pid

            st.success("✅ Patient feedback analyzed successfully! Doctor Clinical Dashboard updated.")
            st.toast("🎉 Analysis complete! Please switch to the Doctor Clinical Analysis Dashboard tab.")

    # Show inline summary if analysis exists
    curr_analysis = st.session_state.get("post_med_analysis")
    if curr_analysis and curr_analysis.get("status") == "SUCCESS":
        st.markdown("### 📊 Immediate Patient Feedback Summary")
        _al = curr_analysis.get("alert_level", "GREEN")
        _badge_col = "#dc2626" if _al == "RED" else ("#f59e0b" if _al == "AMBER" else "#16a34a")

        spoken_txt = curr_analysis.get("spoken_transcript") or tab2_transcript
        transcription_box_html = ""
        if spoken_txt:
            transcription_box_html = (
                f'<div style="background: #eff6ff; border: 1px solid #bfdbfe; border-radius: 6px; padding: 10px 14px; margin: 8px 0 12px 0;">'
                f'<span style="font-weight: 600; color: #1e40af; font-size: 0.9rem;">🎙️ Audio Transcribed Text:</span> '
                f'<span style="color: #1e293b; font-style: italic;">"{spoken_txt}"</span>'
                f'</div>'
            )

        st.markdown(
            f'<div style="border-left: 6px solid {_badge_col}; background: #f8fafc; padding: 14px 18px; border-radius: 8px; margin: 12px 0;">'
            f'<div style="font-weight: 700; color: {_badge_col}; font-size: 1.1rem;">Clinical Assessment: {curr_analysis.get("causality_assessment")}</div>'
            f'{transcription_box_html}'
            f'<div style="font-size: 0.92rem; color: #334155; margin-top: 6px;"><b>Suspected Drug:</b> {curr_analysis.get("suspected_medication")}</div>'
            f'<div style="font-size: 0.92rem; color: #475569; margin-top: 4px;"><b>Reported Symptoms:</b> {", ".join(curr_analysis.get("reported_symptoms", []))}</div>'
            f'<div style="font-size: 0.92rem; color: #0284c7; margin-top: 8px;">👉 <b>Full details, physician recommendations, and emergency warnings are displayed in the 🩺 Doctor Clinical Analysis Dashboard tab.</b></div>'
            f'</div>',
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: DOCTOR CLINICAL ANALYSIS DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown(
        '<div class="sec-head">🩺 Doctor Clinical Analysis Dashboard — Comprehensive Medical Intelligence</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        "Executive Physician Decision Support Command Center. This dashboard synthesizes the "
        "**handwritten prescription extraction**, **patient medical history & allergies**, and **post-medication voice & text feedback** "
        "into actionable clinical guidance, adverse drug reaction (ADR) causality analysis, and safe alternative recommendations."
    )

    rx_res_dash = st.session_state.get("rx_extraction_result")
    post_res_dash = st.session_state.get("post_med_analysis")
    pipeline_res_dash = st.session_state.get("pipeline_result")

    # If neither has been run yet
    if not rx_res_dash and not post_res_dash:
        st.markdown(
            """
            <div style="background: #f8fafc; border: 2px dashed #cbd5e1; border-radius: 16px; padding: 48px 32px; text-align: center; margin: 24px 0;">
                <div style="font-size: 3.2rem; margin-bottom: 16px;">🩺</div>
                <h3 style="color: #334155; margin: 0 0 12px 0;">Doctor Clinical Analysis Dashboard Waiting for Inputs</h3>
                <p style="color: #64748b; font-size: 1rem; margin: 0 auto; max-width: 620px; line-height: 1.6;">
                    To generate the clinical dashboard:
                    <br>1️⃣ Go to <b>📝 Handwritten Prescription & Voice Assistant</b> and run the 3-Stage extraction.
                    <br>2️⃣ Go to <b>💬 Patient Post-Medication Feedback</b> to record or type patient post-visit symptoms.
                    <br>3️⃣ Return here to review the synthesized clinical assessment, ADR causality, and doctor action plan.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        # Determine Overall Alert Level across prescription verification + post-medication feedback
        overall_level = "GREEN"
        triage_message = "Prescription safe. Routine clinical follow-up."
        triage_color = "#16a34a"
        triage_bg = "#f0fdf4"
        triage_icon = "🟢"

        # Check post-medication alert level
        if post_res_dash and post_res_dash.get("status") == "SUCCESS":
            pm_level = post_res_dash.get("alert_level", "GREEN")
            if pm_level == "RED":
                overall_level = "RED"
                triage_message = "CRITICAL ALERT: Suspected Adverse Drug Reaction / Hypersensitivity. Immediate drug discontinuation advised."
                triage_color = "#dc2626"
                triage_bg = "#fef2f2"
                triage_icon = "🔴"
            elif pm_level == "AMBER" and overall_level != "RED":
                overall_level = "AMBER"
                triage_message = "WARNING: Moderate drug intolerance or side effect reported. Dosage adjustment or gastroprotection recommended."
                triage_color = "#f59e0b"
                triage_bg = "#fffbeb"
                triage_icon = "🟠"

        # Check Stage 3 safety pipeline alert level
        if pipeline_res_dash and pipeline_res_dash.get("stage4_doctor_dashboard"):
            s4_level = pipeline_res_dash["stage4_doctor_dashboard"].get("overall_safety_alert_level", "GREEN")
            if s4_level == "RED":
                overall_level = "RED"
                triage_message = "CRITICAL ALERT: Drug-Allergy or Contraindication Conflict identified on prescription."
                triage_color = "#dc2626"
                triage_bg = "#fef2f2"
                triage_icon = "🔴"
            elif s4_level == "AMBER" and overall_level != "RED":
                overall_level = "AMBER"
                triage_message = "WARNING: Potential drug interaction or precaution detected on prescription."
                triage_color = "#f59e0b"
                triage_bg = "#fffbeb"
                triage_icon = "🟠"

        # Top Executive Banner
        st.markdown(
            f"""
            <div style="background:{triage_bg}; border:2px solid {triage_color}; border-radius:14px; padding:22px 26px; margin:14px 0 20px 0;">
                <div style="display:flex; align-items:center; gap:16px;">
                    <div style="font-size:2.8rem;">{triage_icon}</div>
                    <div>
                        <div style="font-size:1.45rem; font-weight:800; color:{triage_color};">OVERALL CLINICAL SAFETY STATUS: {overall_level}</div>
                        <div style="font-size:0.96rem; color:#334155; margin-top:4px; font-weight:500;">{triage_message}</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # 3 KPI Cards
        kpi_c1, kpi_c2, kpi_c3 = st.columns(3)
        with kpi_c1:
            rx_count = len(rx_res_dash.get("medicines", [])) if rx_res_dash else 0
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-val">{rx_count}</div>
                    <div class="kpi-lbl">Prescribed Drugs Digitized</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with kpi_c2:
            sym_count = len(post_res_dash.get("reported_symptoms", [])) if post_res_dash else 0
            st.markdown(
                f"""
                <div class="kpi-card {'amber' if sym_count > 0 else 'green'}">
                    <div class="kpi-val">{sym_count}</div>
                    <div class="kpi-lbl">Reported Adverse Symptoms</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with kpi_c3:
            adr_status = (
                post_res_dash.get("causality_assessment", "None")
                if post_res_dash
                else "No Post-Visit Issue"
            )
            st.markdown(
                f"""
                <div class="kpi-card {'red' if overall_level == 'RED' else ('amber' if overall_level == 'AMBER' else 'green')}">
                    <div class="kpi-val" style="font-size: 1.15rem; font-weight:700; margin-top:6px;">{adr_status}</div>
                    <div class="kpi-lbl">ADR Assessment Status</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.divider()

        # SECTION 1: Digitized Prescription & Patient Intake
        col_p1, col_p2 = st.columns([1.1, 1])

        with col_p1:
            st.markdown("### 📋 1. Digitized Prescription Regimen")
            if rx_res_dash and rx_res_dash.get("status") == "success":
                st.caption(
                    f"Patient: **{rx_res_dash.get('patient_name') or 'N/A'}** | "
                    f"Age/Sex: **{rx_res_dash.get('patient_age') or '—'} yrs / {rx_res_dash.get('patient_gender') or '—'}**"
                )
                if rx_res_dash.get("diagnosed_condition"):
                    st.info(f"🩺 **Clinical Diagnosis:** {rx_res_dash.get('diagnosed_condition')}")
                meds_table_data = []
                for m in rx_res_dash.get("medicines", []):
                    meds_table_data.append({
                        "Prescribed Drug / Active Compound": m.get("medicine_name", ""),
                        "Dosage": m.get("dosage", ""),
                        "Frequency": m.get("frequency", ""),
                        "Duration": m.get("duration", ""),
                        "Certainty": m.get("certainty", ""),
                    })
                if meds_table_data:
                    st.dataframe(pd.DataFrame(meds_table_data), use_container_width=True, hide_index=True)
            else:
                st.caption("No prescription image processed yet. Upload in Tab 1.")

        with col_p2:
            st.markdown("### 👤 2. Pre-Consultation History & Allergies")
            st.markdown("**Documented Drug Allergies:**")
            curr_allergies = current_patient.get("known_allergies", [])
            if curr_allergies:
                for a in curr_allergies:
                    is_none = a in ["None documented", "No known drug allergies", "None"]
                    b_col = "#16a34a" if is_none else "#dc2626"
                    st.markdown(f"- ⚠️ <span style='color:{b_col}; font-weight:600;'>{a}</span>", unsafe_allow_html=True)
            else:
                st.markdown("- *None documented*")

            st.markdown("**Documented Chronic Conditions:**")
            curr_conditions = current_patient.get("known_conditions", [])
            if curr_conditions:
                for c in curr_conditions:
                    st.markdown(f"- 🩺 {c}")
            else:
                st.markdown("- *None documented*")

            # Initial Voice transcript if available
            if pipeline_res_dash and pipeline_res_dash.get("stage3_voice_history"):
                s3_trans = pipeline_res_dash["stage3_voice_history"].get("raw_transcript")
                if s3_trans:
                    with st.expander("🎙️ Pre-Visit Voice Transcript", expanded=False):
                        st.markdown(f"> *{s3_trans}*")

        st.divider()

        # SECTION 2: Post-Medication Patient Symptoms & ADR Causality Analysis
        st.markdown("### 🔍 3. Post-Medication Adverse Drug Reaction (ADR) Causality")

        if post_res_dash and post_res_dash.get("status") == "SUCCESS":
            c_adr1, c_adr2 = st.columns([1, 1])

            with c_adr1:
                st.markdown("#### 🗣️ Patient Reported Experience")
                raw_fb = post_res_dash.get("raw_feedback", "")
                if raw_fb:
                    st.markdown(
                        f"""
                        <div style="background:#f1f5f9; border-left:4px solid #0284c7; padding:12px 16px; border-radius:6px; font-style:italic;">
                            "{raw_fb}"
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                st.markdown("**Extracted Symptoms:**")
                sym_list = post_res_dash.get("reported_symptoms", [])
                if sym_list:
                    for s in sym_list:
                        st.markdown(f"- 🚨 **{s}**")
                else:
                    st.markdown("- *None extracted*")

            with c_adr2:
                st.markdown("#### 🔬 Pharmacological Causality & Mechanism")
                st.markdown(
                    f"**Suspected Offending Drug:** <code style='font-size:1rem; color:#dc2626;'>{post_res_dash.get('suspected_medication', 'None')}</code>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"**Causality Likelihood:** <b>{post_res_dash.get('causality_assessment', 'N/A')}</b>",
                    unsafe_allow_html=True,
                )
                st.markdown("**Biological Mechanism:**")
                st.markdown(
                    f"<p style='font-size:0.92rem; color:#475569;'>{post_res_dash.get('pharmacological_mechanism', '')}</p>",
                    unsafe_allow_html=True,
                )

            st.markdown("---")

            # SECTION 3: Actionable Doctor Recommendations
            st.markdown("### 👨‍⚕️ 4. Actionable Physician Recommendations & Treatment Plan")

            c_rec1, c_rec2 = st.columns(2)

            with c_rec1:
                st.markdown("#### ⚡ Immediate Clinical Interventions")
                doc_recs = post_res_dash.get("doctor_recommendations", [])
                if doc_recs:
                    for idx, r in enumerate(doc_recs, 1):
                        st.markdown(f"**{idx}.** {r}")
                else:
                    st.markdown("- Continue routine therapy under observation.")

                st.markdown("#### 🚨 Emergency Red Flag Warnings")
                r_flags = post_res_dash.get("emergency_red_flags", [])
                if r_flags:
                    for rf in r_flags:
                        st.markdown(f"- ⚠️ <span style='color:#dc2626;'>{rf}</span>", unsafe_allow_html=True)
                else:
                    st.markdown("- No critical emergency red flags reported.")

            with c_rec2:
                st.markdown("#### 💡 Recommended Safe Alternative Medications")
                alts = post_res_dash.get("suggested_alternatives", [])
                if alts:
                    for a in alts:
                        st.markdown(f"- 💊 **{a}**")
                else:
                    st.markdown("- No alternative medication needed at this time.")

                st.markdown("#### 🛡️ Symptomatic Relief Therapies")
                relief = post_res_dash.get("symptomatic_relief", [])
                if relief:
                    for rel in relief:
                        st.markdown(f"- 🩺 {rel}")
                else:
                    st.markdown("- Routine supportive care.")

        else:
            st.info(
                "ℹ️ No post-medication symptoms reported yet. If the patient has taken medication "
                "and experienced an issue, record it in Tab 2 to view ADR causality."
            )

        st.divider()

        # SECTION 4: Export Unified Clinical JSON
        st.markdown("### 💾 5. Export Unified Clinical Intelligence Record")
        st.caption(
            "Export structured record combining digitized prescription, patient profile, and post-medication ADR analysis for EHR integration."
        )

        complete_clinical_record = {
            "patient_profile": {
                "id": selected_pid,
                "name": current_patient.get("name"),
                "age": current_patient.get("age"),
                "gender": current_patient.get("gender"),
                "allergies": current_patient.get("known_allergies", []),
                "conditions": current_patient.get("known_conditions", []),
            },
            "prescription_extraction": rx_res_dash,
            "post_medication_adr_analysis": post_res_dash,
            "overall_safety_alert": overall_level,
            "timestamp": datetime.now().isoformat(),
        }

        st.download_button(
            "⬇️ Download Complete Clinical Intelligence JSON",
            data=json.dumps(complete_clinical_record, indent=2),
            file_name=f"clinical_intelligence_{selected_pid}_{datetime.now().strftime('%Y%m%d%H%M')}.json",
            mime="application/json",
            use_container_width=True,
        )

        st.divider()

        # SECTION 6: Microsoft Power BI Healthcare Analytics Export & Live Data Explorer
        st.markdown("### 📊 6. Microsoft Power BI Healthcare Analytics & Live Data Explorer")
        st.markdown(
            "MedAgent prepares enterprise relational data models structured for **Microsoft Power BI** (`Dim_Patient`, `Dim_Medications`, `Fact_LabResults`, `Fact_Prescriptions`, `Fact_ADR_Causality`). "
            "You can explore the live Power BI-ready tables directly on screen below, inspect the schema, and download individual or batch datasets."
        )

        # Prepare Power BI datasets automatically so judges see live tables immediately
        all_rx_list = current_patient.get("prescriptions", [])
        if rx_res_dash and rx_res_dash.get("status") == "success":
            all_rx_list = [rx_res_dash] + all_rx_list

        all_labs = current_patient.get("lab_reports", [])
        comp_records = impact_analyzer.compare_medication_impact(all_labs)

        # Ensure latest Power BI files are generated in outputs/exports/
        paths = orchestrator.export_longitudinal_powerbi_datasets(
            patient_info=current_patient,
            comparison_records=comp_records,
            prescriptions=all_rx_list,
            lab_reports=all_labs,
        )
        st.session_state["pbi_export_paths"] = paths

        # Top KPI Metrics Ribbon for Judges
        pbi_kpi1, pbi_kpi2, pbi_kpi3, pbi_kpi4 = st.columns(4)
        with pbi_kpi1:
            st.metric("Total Prescriptions Ingested", len(all_rx_list), help="Count of digitized + historical prescriptions")
        with pbi_kpi2:
            st.metric("Biomarker Lab Points", len(all_labs), help="Longitudinal lab tests tracked")
        with pbi_kpi3:
            st.metric("Medication Impact Deltas", len(comp_records), help="Before vs After biomarker parameter transitions")
        with pbi_kpi4:
            st.metric("Data Model Schema", "Kimball Star-Schema", help="Optimized for Power BI / Fabric / Synapse")

        st.markdown("#### 🖥️ Interactive Power BI Data Explorer (Live Relational Tables)")
        st.caption("Inspect each relational table before importing into Power BI Desktop or downloading:")

        pbi_tab1, pbi_tab2, pbi_tab3, pbi_tab4 = st.tabs([
            "📋 1. Dim_Patient (Overview)",
            "💊 2. Fact_Prescriptions (Telemetry)",
            "📈 3. Fact_MedicationImpact (Before/After)",
            "🧪 4. Fact_Biomarkers (Lab History)",
        ])

        with pbi_tab1:
            st.markdown("##### 👤 Dimension: Patient Overview (`Dim_Patient.csv`)")
            df_pt = pd.DataFrame([{
                "Patient_ID": current_patient.get("patient_id", "P001"),
                "Name": current_patient.get("name", "Unknown"),
                "Age": current_patient.get("age", "N/A"),
                "Gender": current_patient.get("gender", "N/A"),
                "Conditions": "; ".join(current_patient.get("known_conditions", [])),
                "Allergies": "; ".join(current_patient.get("known_allergies", [])),
                "Adverse_Reactions": "; ".join(current_patient.get("adverse_reactions", [])),
                "Current_Medications": "; ".join(current_patient.get("current_medications", [])),
            }])
            st.dataframe(df_pt, use_container_width=True, hide_index=True)
            if Path(paths.get("patient_overview", "")).exists():
                st.download_button(
                    "⬇️ Download Dim_Patient.csv",
                    data=Path(paths["patient_overview"]).read_bytes(),
                    file_name="Dim_Patient.csv",
                    mime="text/csv",
                    key="dl_pbi_pt_inline",
                    use_container_width=False,
                )

        with pbi_tab2:
            st.markdown("##### 💊 Fact: Prescription Telemetry (`Fact_Prescriptions.csv`)")
            rx_rows_display = []
            for rx in all_rx_list:
                rx_d = rx.get("date", "")
                dr = rx.get("doctor_name", "")
                clinic = rx.get("clinic_name", "")
                cond = rx.get("diagnosed_condition", "")
                for med in rx.get("medicines", []):
                    rx_rows_display.append({
                        "Prescription_Date": rx_d,
                        "Doctor": dr,
                        "Clinic": clinic,
                        "Diagnosed_Condition": cond,
                        "Medicine_Name": med.get("medicine_name", ""),
                        "Form": med.get("form", "Tablet"),
                        "Dosage": med.get("dosage", ""),
                        "Frequency": med.get("frequency", ""),
                        "Duration": med.get("duration", ""),
                        "Certainty": med.get("certainty", "High"),
                    })
            df_rx_disp = pd.DataFrame(rx_rows_display) if rx_rows_display else pd.DataFrame(columns=["Prescription_Date", "Medicine_Name", "Dosage", "Frequency"])
            st.dataframe(df_rx_disp, use_container_width=True, hide_index=True)
            if Path(paths.get("prescriptions", "")).exists():
                st.download_button(
                    "⬇️ Download Fact_Prescriptions.csv",
                    data=Path(paths["prescriptions"]).read_bytes(),
                    file_name="Fact_Prescriptions.csv",
                    mime="text/csv",
                    key="dl_pbi_rx_inline",
                    use_container_width=False,
                )

        with pbi_tab3:
            st.markdown("##### 📈 Fact: Medication Impact Longitudinal Deltas (`Fact_MedicationImpact.csv`)")
            df_comp_disp = pd.DataFrame(comp_records) if comp_records else pd.DataFrame(columns=["Parameter", "Date of Test", "Previous Value", "Current Value", "Difference / Change", "Percentage Change", "Status Indicator"])
            clean_comp_cols = [c for c in df_comp_disp.columns if not c.startswith("raw_")]
            st.dataframe(df_comp_disp[clean_comp_cols] if not df_comp_disp.empty else df_comp_disp, use_container_width=True, hide_index=True)
            if Path(paths.get("medication_impact", "")).exists():
                st.download_button(
                    "⬇️ Download Fact_MedicationImpact.csv",
                    data=Path(paths["medication_impact"]).read_bytes(),
                    file_name="Fact_MedicationImpact.csv",
                    mime="text/csv",
                    key="dl_pbi_comp_inline",
                    use_container_width=False,
                )

        with pbi_tab4:
            st.markdown("##### 🧪 Fact: Biomarker Lab History (`Fact_Biomarkers.csv`)")
            df_labs_disp = pd.DataFrame(all_labs) if all_labs else pd.DataFrame(columns=["date", "parameter", "value", "unit", "reference_range", "status"])
            st.dataframe(df_labs_disp, use_container_width=True, hide_index=True)
            if Path(paths.get("biomarkers", "")).exists():
                st.download_button(
                    "⬇️ Download Fact_Biomarkers.csv",
                    data=Path(paths["biomarkers"]).read_bytes(),
                    file_name="Fact_Biomarkers.csv",
                    mime="text/csv",
                    key="dl_pbi_labs_inline",
                    use_container_width=False,
                )

        st.markdown("---")
        # Collapsible Guide: How to demonstrate in Power BI
        with st.expander("💡 Judge Demonstration Guide: How Power BI Connects to MedAgent", expanded=True):
            st.markdown(
                """
                ### 🎙️ What to tell the Judges:
                > *"MedAgent doesn't just stop at digitizing handwritten prescriptions — it continuously feeds an enterprise-ready **Star-Schema relational data warehouse** for clinical business intelligence in Microsoft Power BI. Hospital chief medical officers can track prescription volumes, drug-drug interaction alerts, and longitudinal patient health outcomes across multiple visits in real-time."*

                ### 🖥️ Why does a `.csv` open in Excel by default?
                - A `.csv` (Comma-Separated Values) file is the **standard universal ingestion format** for Power BI, Tableau, and SQL data warehouses.
                - Because Microsoft Windows associates `.csv` files with Microsoft Excel by default, double-clicking any CSV file opens Excel.
                - **To show Power BI directly to the judges:**
                  1. Launch **Power BI Desktop** on your computer.
                  2. Click **Get Data** &rarr; **Text/CSV** (or **Folder**).
                  3. Select the folder: `outputs/exports`.
                  4. Power BI will auto-detect the relationships (`Dim_Patient` &harr; `Fact_Prescriptions` &harr; `Fact_MedicationImpact`)!
                """
            )

        col_pbi_dl1, col_pbi_dl2 = st.columns([1, 1])
        with col_pbi_dl1:
            st.markdown("#### 🌟 Relational Schema Architecture")
            st.markdown(
                """
                - **`Dim_Patient.csv`**: Patient demographics, allergies, chronic conditions, and past ADRs.
                - **`Fact_Prescriptions.csv`**: Full stroke OCR telemetry, certainty scores, doses, and directions.
                - **`Fact_MedicationImpact.csv`**: Before vs after biomarker delta, percentage shifts, and efficacy.
                - **`Fact_Biomarkers.csv`**: Longitudinal lab tests (HbA1c, Fasting Glucose, eGFR, BP).
                """
            )
        with col_pbi_dl2:
            st.markdown("#### 🚀 Batch Export All Datasets")
            st.caption("Export all 4 tables at once to `outputs/exports/` for direct Power BI Desktop loading.")
            if st.button("🔄 Re-sync All Datasets Now", type="secondary", use_container_width=True):
                paths_fresh = orchestrator.export_longitudinal_powerbi_datasets(
                    patient_info=current_patient,
                    comparison_records=comp_records,
                    prescriptions=all_rx_list,
                    lab_reports=all_labs,
                )
                st.session_state["pbi_export_paths"] = paths_fresh
                st.success(f"✅ All 4 Power BI datasets refreshed in `outputs/exports/` at {datetime.now().strftime('%H:%M:%S')}!")
