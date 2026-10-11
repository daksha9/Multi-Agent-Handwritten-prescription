"""Multi-Agent Intelligence for Handwritten Prescription and Predictive Health Analytics.

Objective 1: Multimodal Vision AI Handwritten Prescription Extraction (Gemini Vision)
Objective 2: Patient Report Comparison & Medication Impact Analysis
Objective 3: Interactive Power BI Healthcare Analytics Dashboard
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
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
with st.sidebar:
    st.image("https://img.icons8.com/color/96/hospital-2.png", width=56)
    st.markdown("## 🏥 Patient Workspace")
    st.caption("Active longitudinal patient profile loaded into memory:")

    patient_summaries = db.list_patients()
    patient_options = {
        p["patient_id"]: f"{p['name']} ({p['patient_id']} - {p['age']}y/{p['gender']})" for p in patient_summaries
    }
    patient_options["BLANK_NEW"] = "➕ New / Blank Patient Profile (No Data Loaded)"

    selected_pid_raw = st.selectbox(
        "Select Patient Profile:",
        options=list(patient_options.keys()),
        format_func=lambda x: patient_options[x],
        index=len(patient_options) - 1,
    )

    is_blank_profile = selected_pid_raw == "BLANK_NEW"

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
        selected_pid: str = str(selected_pid_raw) if selected_pid_raw else "P001"
        current_patient = db.get_patient(selected_pid) or {}

    st.divider()
    st.markdown("### 👤 Patient Clinical Summary")
    if is_blank_profile:
        st.markdown(
            '<div style="padding: 12px 14px; background: #f8fafc; border: 1px dashed #cbd5e1; border-radius: 10px; margin-bottom: 14px; line-height: 1.45;">'
            '<span style="color: #64748b; font-size: 0.85rem;">📋 No patient data loaded. Upload a prescription in Tab 1 or select an existing patient to populate this profile.</span>'
            '</div>'
            '<div style="padding: 4px 4px 8px 8px;">'
            '  <div style="margin: 6px 0;"><strong style="font-size: 0.9rem;">Name:</strong> <span style="color: #94a3b8;">— Not loaded —</span></div>'
            '  <div style="margin: 6px 0;"><strong style="font-size: 0.9rem;">Age / Sex:</strong> <span style="color: #94a3b8;">— — / —</span></div>'
            '  <div style="margin: 10px 0 4px;"><strong style="font-size: 0.9rem;">Known Medical Conditions:</strong></div>'
            '  <div style="margin: 0 0 10px 6px;"><span style="color: #94a3b8;">· No documented conditions</span></div>'
            '  <div style="margin: 10px 0 4px;"><strong style="font-size: 0.9rem;">Documented Drug Allergies:</strong></div>'
            '  <div style="margin: 0 0 10px 6px;"><span style="color: #94a3b8;">· No documented allergies</span></div>'
            '  <div style="margin: 10px 0 4px;"><strong style="font-size: 0.9rem;">Reported Adverse Reactions:</strong></div>'
            '  <div style="margin: 0 0 10px 6px;"><span style="color: #94a3b8;">· No adverse reactions recorded</span></div>'
            '  <div style="margin: 10px 0 4px;"><strong style="font-size: 0.9rem;">Safety Measures & Alerts:</strong></div>'
            '  <div style="margin: 0 0 4px 6px;"><span style="color: #94a3b8;">· No active safety alerts</span></div>'
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

    # Interactive Clinical Profile Editor (hidden for blank profiles)
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
    else:
        with st.expander("➕ Create New Patient Profile", expanded=False):
            st.caption("Create a new patient record to begin working.")
            st.markdown(
                '<div style="padding: 0 6px;">',
                unsafe_allow_html=True,
            )
            new_patient_name = st.text_input("Patient Full Name:", placeholder="e.g. John Michael Smith")
            new_patient_age = st.number_input("Age:", min_value=0, max_value=130, value=30, step=1)
            new_patient_gender = st.selectbox("Gender:", ["Male", "Female", "Other"], index=0)
            st.markdown('</div>', unsafe_allow_html=True)
            if st.button("➕ Create New Patient", type="primary", use_container_width=True):
                if new_patient_name and new_patient_name.strip():
                    new_pid = f"P{len(db.list_patients()) + 1:03d}"
                    db.add_or_update_patient({
                        "patient_id": new_pid,
                        "name": new_patient_name.strip(),
                        "age": int(new_patient_age),
                        "gender": new_patient_gender,
                        "known_conditions": [],
                        "known_allergies": ["No known drug allergies"],
                        "adverse_reactions": ["None"],
                        "medical_history": "New patient record created.",
                        "current_medications": [],
                        "medication_timeline": [],
                        "prescriptions": [],
                        "lab_reports": [],
                    })
                    st.success(f"✅ Patient {new_patient_name.strip()} ({new_pid}) created! Please select from dropdown.")
                    st.toast("New patient profile created.")
                    st.rerun()
                else:
                    st.warning("⚠️ Please enter a patient name.")

    st.divider()
    st.markdown("""
    **Architecture Core:**
    - 👁️ **Objective 1:** Multimodal Gemini Vision AI
    - 📈 **Objective 2:** Medication Impact & Report Trends
    - 📊 **Objective 3:** Power BI Healthcare Dashboard
    """)
    st.caption("Multi-Agent Intelligence for Handwritten Prescription and Predictive Health Analytics")


# ══════════════════════════════════════════════════════════════════════════════
# HERO HEADER BANNER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
    """
<div class="hero-banner">
    <div class="hero-title">🏥 Multi-Agent Intelligence for Handwritten Prescription and Predictive Health Analytics</div>
    <div class="hero-sub">
        End-to-End Multimodal Vision AI Handwritten Prescription Extraction (Gemini Vision) &nbsp;·&nbsp;
        Longitudinal Patient Report Comparison &nbsp;·&nbsp; Before/After Medication Impact Trajectory &nbsp;·&nbsp;
        Interactive Power BI Healthcare Dashboard
    </div>
    <div class="hero-pills">
        <span class="hero-pill">👁️ Multimodal Gemini Vision AI</span>
        <span class="hero-pill">📈 Chronological Lab Impact Tracking</span>
        <span class="hero-pill">📊 Interactive Power BI Visualizations</span>
        <span class="hero-pill">🛡️ Dynamic Allergy & Safety Conflict Matrix</span>
        <span class="hero-pill">⚖️ Evidence-Based Descriptive Trend Analytics</span>
    </div>
</div>
""",
    unsafe_allow_html=True,
)


tab1, tab2, tab3 = st.tabs(
    [
        "📝 Handwritten Prescription",
        "📈 Reports Comparision",
        "📊 Report Analysis",
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
            st.image(rx_image, caption="Uploaded Prescription Pad", use_container_width=True)

        with col_act:
            st.markdown("### 🤖 Extraction Configuration")
            if vision_extractor.is_available:
                st.success(
                    "✅ **Multimodal Gemini Vision AI is Online**\n"
                    "Primary Model: `gemini-3.5-flash` with zero offline OCR fallback."
                )
            else:
                st.warning("⚠️ Gemini Vision API key missing. Please configure GEMINI_API_KEY in `config/.env`.")

            st.info(
                "💡 **Clinical Safety Policy:** Gemini Vision validates medicine names against pharmacological databases "
                "and flags uncertain strokes rather than hallucinating."
            )

            run_extract = st.button(
                "🚀 Run Gemini Vision Prescription Extraction", type="primary", use_container_width=True
            )
            if run_extract:
                progress_bar = st.progress(0, text="Initializing Gemini Vision AI pipeline...")
                time.sleep(0.2)
                progress_bar.progress(
                    25, text="🔍 Step 1/4: Analyzing image raster & pre-conditioning handwriting strokes..."
                )
                time.sleep(0.3)
                progress_bar.progress(
                    55, text="🧠 Step 2/4: Deciphering messy doctor cursive script via Multimodal Gemini Vision..."
                )

                ext_result = vision_extractor.extract(rx_image)

                progress_bar.progress(
                    80, text="💊 Step 3/4: Parsing clinical entities (Medicines, Dosages, Frequencies, Instructions)..."
                )
                time.sleep(0.2)
                progress_bar.progress(
                    100, text="✅ Step 4/4: Cross-verifying against pharmacological database & flagging certainty..."
                )
                time.sleep(0.2)
                progress_bar.empty()

                st.session_state["rx_extraction_result"] = ext_result
                st.toast("Prescription extraction completed!")

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

        # Patient Profile Synchronization Card (Linking Tab 1 to Tab 2)
        st.markdown("#### 🔄 Update Patient Information from Prescription (Link to Reports Comparision)")
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
                    value=rx_res.get("patient_name") or current_patient.get("name", "John Doe"),
                )
            with col_p2:
                default_age = rx_res.get("patient_age")
                try:
                    default_age_val = int(default_age) if default_age else int(current_patient.get("age", 45))
                except Exception:
                    default_age_val = int(current_patient.get("age", 45))
                sync_age = st.number_input("Age:", value=default_age_val, min_value=1, max_value=120)
            with col_p3:
                gender_opts = ["Male", "Female", "Other"]
                curr_g = rx_res.get("patient_gender") or current_patient.get("gender", "Male")
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
                f"Prescription stored and linked to **Reports Comparision** tab."
            )
            st.toast("Patient profile updated & linked to Reports Comparision!")
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
# TAB 2: OBJECTIVE 2 — PATIENT REPORT COMPARISON & MEDICATION IMPACT ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown(
        '<div class="sec-head">📈 Reports Comparision — Sequential Report Upload & Medication Impact Analysis</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        "Upload multi-stage laboratory test reports and associate each report with its corresponding medication "
        "(e.g., **Report 1 with Medicine A** $\\rightarrow$ **Report 2 with Medicine B**). "
        "The system chronologically compares the before-medication and after-medication values, tracks percentage changes, "
        "identifies whether patient condition worsened, improved, or stabilized, and plots medication impact trajectories."
    )

    if is_blank_profile:
        st.markdown(
            """
            <div style="padding: 14px 18px; background: #fef3c7; border: 1px solid #fbbf24; border-left: 5px solid #d97706; border-radius: 10px; margin: 14px 0 18px 0;">
                <div style="font-weight: 700; color: #78350f; margin-bottom: 6px;">🚫 Save Disabled — No Patient Profile Selected</div>
                <div style="font-size: 0.9rem; color: #78350f; line-height: 1.45;">
                    Laboratory reports must be linked to a specific patient. All <b>Save buttons below are disabled</b> until you:
                    <ol style="margin: 6px 0 4px 20px;">
                        <li>Use the sidebar ➕ <b>Create New Patient Profile</b> expander to set up a record, OR</li>
                        <li>Select an existing patient (e.g., John Doe / P001) from the sidebar dropdown.</li>
                    </ol>
                    You can still pre-fill the biomarker number inputs below — they will be retained when you select a patient.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    lab_reports = [] if is_blank_profile else current_patient.get("lab_reports", [])
    med_timeline = [] if is_blank_profile else current_patient.get("medication_timeline", [])

    # ──────────────────────────────────────────────────────────────────────────
    # SEQUENTIAL REPORT UPLOAD & MEDICATION ASSOCIATION (OBJECTIVE 2 WORKFLOW)
    # ──────────────────────────────────────────────────────────────────────────
    st.markdown("### 📤 Sequential Report Upload & Medication Association")
    st.caption(
        "Follow the sequential workflow: upload the initial baseline report with its medicine name, "
        "then upload the follow-up report with the new medicine name to measure therapeutic impact."
    )

    col_rep1, col_rep2 = st.columns(2)

    with col_rep1:
        st.markdown(
            """
            <div style="background: #f8fafc; border: 1px solid #cbd5e1; border-top: 4px solid #0284c7; border-radius: 12px; padding: 16px; margin-bottom: 16px;">
                <h4 style="margin: 0 0 8px 0; color: #0f172a;">🧪 Step 1: Initial Medical Report & Medicine A</h4>
                <p style="font-size: 0.84rem; color: #64748b; margin-bottom: 12px;">Baseline laboratory workup prior to or during initial medication regimen.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        up_rep1 = st.file_uploader(
            "Upload Report 1 (PNG, JPG, PDF, CSV):",
            type=["png", "jpg", "jpeg", "pdf", "csv"],
            key="uploader_report_1",
            help="Initial / baseline laboratory report file.",
        )
        med1_default = st.session_state.get("tab1_linked_med") or "Medicine A: Metformin 500mg BD"
        med1_name = st.text_input(
            "Medicine Name for Report 1:",
            value=med1_default,
            key="rep1_med_name",
            help="E.g., Medicine A, Metformin 500mg BD, or Lifestyle observation",
        )
        date1_val = st.date_input("Report 1 Test Date:", value=datetime(2024, 11, 15), key="rep1_date")

        st.write("**Key Biomarker Observations for Report 1:**")
        c1_p1, c1_p2 = st.columns(2)
        with c1_p1:
            hba1c_1 = st.number_input("HbA1c (%):", value=8.4, step=0.1, key="rep1_hba1c")
            sbp_1 = st.number_input("Systolic BP (mmHg):", value=142.0, step=1.0, key="rep1_sbp")
        with c1_p2:
            fbg_1 = st.number_input("Fasting Glucose (mg/dL):", value=165.0, step=1.0, key="rep1_fbg")
            chol_1 = st.number_input("Total Cholesterol (mg/dL):", value=235.0, step=1.0, key="rep1_chol")

        if st.button("💾 Save Report 1 & Medicine A", type="primary", use_container_width=True, key="btn_save_rep1", disabled=is_blank_profile):
            try:
                db.add_medication_period(
                    selected_pid,
                    period_name=f"Period 1: {med1_name}",
                    start_date=date1_val.strftime("%Y-%m-%d"),
                    end_date="2025-02-14",
                    medications=[med1_name],
                    clinical_target="Initial diagnostic & therapeutic baseline",
                )
                entries1 = [
                    {
                        "date": date1_val.strftime("%Y-%m-%d"),
                        "parameter": "HbA1c",
                        "value": float(hba1c_1),
                        "unit": "%",
                        "reference_range": "4.0 - 5.6",
                        "condition": "Type 2 Diabetes",
                        "associated_medication": med1_name,
                    },
                    {
                        "date": date1_val.strftime("%Y-%m-%d"),
                        "parameter": "Fasting Blood Glucose",
                        "value": float(fbg_1),
                        "unit": "mg/dL",
                        "reference_range": "70.0 - 99.0",
                        "condition": "Type 2 Diabetes",
                        "associated_medication": med1_name,
                    },
                    {
                        "date": date1_val.strftime("%Y-%m-%d"),
                        "parameter": "Systolic Blood Pressure",
                        "value": float(sbp_1),
                        "unit": "mmHg",
                        "reference_range": "90.0 - 120.0",
                        "condition": "Hypertension",
                        "associated_medication": med1_name,
                    },
                    {
                        "date": date1_val.strftime("%Y-%m-%d"),
                        "parameter": "Total Cholesterol",
                        "value": float(chol_1),
                        "unit": "mg/dL",
                        "reference_range": "125.0 - 200.0",
                        "condition": "Dyslipidemia",
                        "associated_medication": med1_name,
                    },
                ]
                for ent in entries1:
                    db.add_lab_report(selected_pid, ent)
                st.success("✅ Report 1 & Medicine A recorded successfully!")
                st.toast("Report 1 saved!")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Failed to save Report 1: {str(e)}")

    with col_rep2:
        st.markdown(
            """
            <div style="background: #f8fafc; border: 1px solid #cbd5e1; border-top: 4px solid #16a34a; border-radius: 12px; padding: 16px; margin-bottom: 16px;">
                <h4 style="margin: 0 0 8px 0; color: #0f172a;">🔬 Step 2: Follow-up Medical Report & Medicine B</h4>
                <p style="font-size: 0.84rem; color: #64748b; margin-bottom: 12px;">Follow-up laboratory workup after patient initiates or shifts to new medication.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        up_rep2 = st.file_uploader(
            "Upload Report 2 (PNG, JPG, PDF, CSV):",
            type=["png", "jpg", "jpeg", "pdf", "csv"],
            key="uploader_report_2",
            help="Follow-up laboratory report file.",
        )
        med2_name = st.text_input(
            "Medicine Name for Report 2:",
            value="Medicine B: Metformin 1000mg + Glimepiride 1mg",
            key="rep2_med_name",
            help="E.g., Medicine B, Metformin 1000mg + Glimepiride 1mg, or Glipizide 5mg",
        )
        date2_val = st.date_input("Report 2 Test Date:", value=datetime(2025, 2, 15), key="rep2_date")

        st.write("**Key Biomarker Observations for Report 2:**")
        c2_p1, c2_p2 = st.columns(2)
        with c2_p1:
            hba1c_2 = st.number_input("HbA1c (%):", value=7.1, step=0.1, key="rep2_hba1c")
            sbp_2 = st.number_input("Systolic BP (mmHg):", value=126.0, step=1.0, key="rep2_sbp")
        with c2_p2:
            fbg_2 = st.number_input("Fasting Glucose (mg/dL):", value=128.0, step=1.0, key="rep2_fbg")
            chol_2 = st.number_input("Total Cholesterol (mg/dL):", value=195.0, step=1.0, key="rep2_chol")

        if st.button("💾 Save Report 2 & Medicine B", type="primary", use_container_width=True, key="btn_save_rep2", disabled=is_blank_profile):
            try:
                db.add_medication_period(
                    selected_pid,
                    period_name=f"Period 2: {med2_name}",
                    start_date=date2_val.strftime("%Y-%m-%d"),
                    end_date="2025-09-30",
                    medications=[med2_name],
                    clinical_target="Subsequent therapy response evaluation",
                )
                entries2 = [
                    {
                        "date": date2_val.strftime("%Y-%m-%d"),
                        "parameter": "HbA1c",
                        "value": float(hba1c_2),
                        "unit": "%",
                        "reference_range": "4.0 - 5.6",
                        "condition": "Type 2 Diabetes",
                        "associated_medication": med2_name,
                    },
                    {
                        "date": date2_val.strftime("%Y-%m-%d"),
                        "parameter": "Fasting Blood Glucose",
                        "value": float(fbg_2),
                        "unit": "mg/dL",
                        "reference_range": "70.0 - 99.0",
                        "condition": "Type 2 Diabetes",
                        "associated_medication": med2_name,
                    },
                    {
                        "date": date2_val.strftime("%Y-%m-%d"),
                        "parameter": "Systolic Blood Pressure",
                        "value": float(sbp_2),
                        "unit": "mmHg",
                        "reference_range": "90.0 - 120.0",
                        "condition": "Hypertension",
                        "associated_medication": med2_name,
                    },
                    {
                        "date": date2_val.strftime("%Y-%m-%d"),
                        "parameter": "Total Cholesterol",
                        "value": float(chol_2),
                        "unit": "mg/dL",
                        "reference_range": "125.0 - 200.0",
                        "condition": "Dyslipidemia",
                        "associated_medication": med2_name,
                    },
                ]
                for ent in entries2:
                    db.add_lab_report(selected_pid, ent)
                st.success("✅ Report 2 & Medicine B recorded successfully!")
                st.toast("Report 2 saved & medication impact calculated!")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Failed to save Report 2: {str(e)}")

    # Expander to add subsequent reports (Report 3, 4...) & medicine name
    with st.expander("➕ Add Subsequent Medical Report (Report 3, 4...) & New Medicine Name", expanded=False):
        patient_display_name = current_patient.get("name") or "Selected Patient"
        st.write(
            f"If the physician changed medication again (e.g., to Medicine C), add another sequential observation for **{patient_display_name}**:"
        )
        if is_blank_profile:
            st.info("ℹ️ Select or create a patient first to enable this section's Save button.")
        with st.form("form_add_subsequent_lab"):
            c_s1, c_s2, c_s3 = st.columns(3)
            with c_s1:
                sub_param_name = st.selectbox(
                    "Parameter Name:",
                    list(impact_analyzer.PARAMETER_METRICS.keys()),
                    index=0,
                    key="sub_param_name",
                )
                sub_metric_def = impact_analyzer.PARAMETER_METRICS.get(sub_param_name, {})
            with c_s2:
                sub_val = st.number_input("Observed Value:", value=6.2, step=0.1, key="sub_val")
                sub_unit = st.text_input("Unit:", value=sub_metric_def.get("unit", ""), key="sub_unit")
            with c_s3:
                sub_date = st.date_input("Test Date:", key="sub_date")
                sub_ref = st.text_input(
                    "Reference Range:",
                    value=f"{sub_metric_def.get('normal_min', '')} - {sub_metric_def.get('normal_max', '')}",
                    key="sub_ref",
                )

            sub_med = st.text_input(
                "Medicine Name for this Subsequent Report:",
                value="Medicine C: Metformin 1000mg + Empagliflozin 10mg",
                key="sub_med_name",
            )

            submit_sub = st.form_submit_button("💾 Append Subsequent Report", type="primary")
            if submit_sub:
                if is_blank_profile:
                    st.warning("⚠️ Please select or create a patient from the sidebar before appending subsequent reports.")
                else:
                    try:
                        sub_entry = {
                            "date": sub_date.strftime("%Y-%m-%d"),
                            "parameter": sub_param_name,
                            "value": float(sub_val),
                            "unit": sub_unit,
                            "reference_range": sub_ref,
                            "condition": sub_metric_def.get("condition", "General Health"),
                            "associated_medication": sub_med,
                        }
                        db.add_lab_report(selected_pid, sub_entry)
                        st.success("✅ Subsequent report recorded! Trajectory updated.")
                        st.toast("Subsequent report recorded.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Failed to append subsequent report: {str(e)}")

    st.divider()

    # ──────────────────────────────────────────────────────────────────────────
    # BEFORE / AFTER PARAMETER COMPARISON TABLE (10 COLUMNS REQUIRED)
    # ──────────────────────────────────────────────────────────────────────────
    st.markdown("### 📋 Before/After Parameter Comparison Table")
    st.caption(
        "Compares before-medication (Report 1) and after-medication (Report 2) values, showing the difference, percentage change, "
        "associated medication shift, and descriptive interpretation of clinical trajectory."
    )

    comparison_data = impact_analyzer.compute_parameter_comparison_table(lab_reports, med_timeline)
    if comparison_data:
        df_comp_display = pd.DataFrame(comparison_data)[
            [
                "Parameter",
                "Date of Test",
                "Previous Value",
                "Current Value",
                "Unit",
                "Reference Normal Range",
                "Difference / Change",
                "Percentage Change",
                "Associated Medication",
                "Condition Monitored",
                "Trend Interpretation",
            ]
        ]
        st.dataframe(df_comp_display, use_container_width=True, hide_index=True)
    else:
        st.info("No comparative laboratory reports recorded yet. Upload Report 1 and Report 2 above.")

    st.divider()

    # ──────────────────────────────────────────────────────────────────────────
    # PARAMETER TIME-SERIES VISUALIZER WITH SHADED MEDICATION BANDS
    # ──────────────────────────────────────────────────────────────────────────
    st.markdown("### 📊 Longitudinal Parameter Trajectory & Medication Periods")
    st.caption(
        "Time-series chart tracking biomarker trajectory across medication periods with shaded reference range and medication transition bands."
    )

    if lab_reports:
        available_params = sorted({r["parameter"] for r in lab_reports})
        col_sel_p, col_info_p = st.columns([1.5, 2.5])

        with col_sel_p:
            selected_param: str = str(
                st.selectbox("Select Parameter to Plot:", available_params, index=0, key="plot_param_sel")
                or (available_params[0] if available_params else "HbA1c")
            )

        with col_info_p:
            p_metric = impact_analyzer.PARAMETER_METRICS.get(selected_param, {})
            st.caption(
                f"**Target Condition:** {p_metric.get('condition', 'General Health')} | "
                f"**Normal Clinical Reference Range:** {p_metric.get('normal_min', 'N/A')} - {p_metric.get('normal_max', 'N/A')} {p_metric.get('unit', '')}"
            )

        df_param = impact_analyzer.get_time_series_data_by_parameter(lab_reports, selected_param)

        if not df_param.empty:
            fig_ts = go.Figure()

            # Normal Range Shaded Green Band
            norm_min = p_metric.get("normal_min")
            norm_max = p_metric.get("normal_max")
            if norm_min is not None and norm_max is not None:
                fig_ts.add_hrect(
                    y0=norm_min,
                    y1=norm_max,
                    fillcolor="rgba(34, 197, 94, 0.12)",
                    line_width=0,
                    annotation_text=f"Normal Target Range ({norm_min} - {norm_max})",
                    annotation_position="bottom right",
                )

            # Observed Lab Points & Line
            fig_ts.add_trace(
                go.Scatter(
                    x=df_param["date_str"],
                    y=df_param["value"],
                    mode="lines+markers+text",
                    name=selected_param,
                    line={"color": "#0284c7", "width": 3},
                    marker={"size": 10, "color": "#0369a1", "line": {"color": "white", "width": 2}},
                    text=[f"{v:.1f}" for v in df_param["value"]],
                    textposition="top center",
                    hovertemplate="<b>Date:</b> %{x}<br><b>Value:</b> %{y} "
                    + p_metric.get("unit", "")
                    + "<extra></extra>",
                )
            )

            # Add Vertical / Shaded Medication Phase Dividers
            period_colors = [
                "rgba(238, 242, 255, 0.5)",
                "rgba(254, 243, 199, 0.45)",
                "rgba(240, 253, 244, 0.5)",
                "rgba(253, 242, 248, 0.5)",
            ]
            for idx, period in enumerate(med_timeline):
                p_start = period.get("start_date")
                p_end = period.get("end_date")

                fig_ts.add_vrect(
                    x0=p_start,
                    x1=p_end,
                    fillcolor=period_colors[idx % len(period_colors)],
                    line={"color": "rgba(100, 116, 139, 0.3)", "width": 1, "dash": "dot"},
                    annotation_text=f"<b>{period.get('period_id')}</b>",
                    annotation_position="top left",
                    layer="below",
                )

            fig_ts.update_layout(
                title=f"Chronological Trajectory: {selected_param} across Medication Periods",
                xaxis_title="Date of Lab Report",
                yaxis_title=f"{selected_param} ({p_metric.get('unit', '')})",
                height=420,
                margin={"l": 20, "r": 20, "t": 50, "b": 30},
                plot_bgcolor="white",
                paper_bgcolor="white",
                font={"family": "Inter", "size": 12},
            )
            fig_ts.update_xaxes(showgrid=True, gridcolor="#f1f5f9")
            fig_ts.update_yaxes(showgrid=True, gridcolor="#f1f5f9")
            st.plotly_chart(fig_ts, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: REPORT ANALYSIS — POWER BI HEALTHCARE DASHBOARD & PREDICTIVE ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown(
        '<div class="sec-head">📊 Report Analysis — Power BI Healthcare Dashboard & Predictive Health Analytics</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        "Interactive Power BI Healthcare Analytics Command Center combining longitudinal report analysis, "
        "medication regimen roadmaps, and **Predictive Health Analytics** for biomarker trajectory forecasting, "
        "time-to-normalcy estimation, and clinical risk stratification. All visual analytics are exportable for **Power BI Desktop**."
    )

    if is_blank_profile:
        st.markdown(
            """
            <div style="background: #f8fafc; border: 2px dashed #cbd5e1; border-radius: 16px; padding: 48px 32px; text-align: center; margin: 24px 0;">
                <div style="font-size: 3.2rem; margin-bottom: 16px;">📊</div>
                <h3 style="color: #334155; margin: 0 0 12px 0;">Report Analysis Dashboard Inactive</h3>
                <p style="color: #64748b; font-size: 1rem; margin: 0 auto; max-width: 620px; line-height: 1.6;">
                    No patient profile and no laboratory reports are currently loaded.
                    <br><br>
                    <b>To begin Report Analysis:</b>
                    <br>1️⃣ Select an existing patient from the sidebar dropdown
                    <br>2️⃣ Or upload sequential reports in the <b>Reports Comparision</b> tab
                    <br>3️⃣ Return here to view longitudinal analytics, KPIs, and forecasting
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        comparison_data = impact_analyzer.compute_parameter_comparison_table(lab_reports, med_timeline)

        # ── Interactive Dashboard Filters (always rendered in scope; type-checker sees ONE canonical assignment) ──
        st.markdown("#### 🎛️ Dynamic Clinical Filters")
        flt_col1, flt_col2, flt_col3 = st.columns([1.2, 1.2, 1.6])

        all_conditions_in_patient: list[str] = []
        all_meds_in_patient: list[str] = []
        all_biomarkers: list[str] = []
        if lab_reports:
            all_conditions_in_patient = sorted({r.get("condition", "General Health") for r in lab_reports if r.get("condition")})
            all_meds_in_patient = sorted({r.get("associated_medication", "General Care") for r in lab_reports if r.get("associated_medication")})
            all_biomarkers = sorted({r["parameter"] for r in lab_reports})

        with flt_col1:
            cond_filter = st.selectbox(
                "Filter by Condition:",
                options=["All Conditions", *all_conditions_in_patient],
                index=0,
                disabled=not lab_reports,
                help="Filter analytics to a specific clinical condition",
            )
        with flt_col2:
            med_filter = st.selectbox(
                "Filter by Medication:",
                options=["All Medications", *all_meds_in_patient],
                index=0,
                disabled=not lab_reports,
                help="Filter analytics to an associated medication period",
            )
        with flt_col3:
            default_bio = all_biomarkers[:4] if len(all_biomarkers) >= 4 else all_biomarkers
            selected_biomarkers: list[str] = list(
                st.multiselect(
                    "Filter Biomarkers for Visual Analytics:",
                    options=all_biomarkers,
                    default=default_bio,
                    disabled=not lab_reports,
                )
            )

        # Single canonical computation of filtered_reports (dominates all downstream usage sites)
        filtered_reports: list[dict[str, Any]] = list(lab_reports)
        if lab_reports:
            if cond_filter != "All Conditions":
                filtered_reports = [r for r in filtered_reports if r.get("condition") == cond_filter]
            if med_filter != "All Medications":
                filtered_reports = [r for r in filtered_reports if r.get("associated_medication") == med_filter]

        st.write("")

        if not lab_reports:
            st.info(
                "💡 No laboratory reports are currently recorded for this patient. "
                "Navigate to the **Reports Comparision** tab to upload and associate baseline and follow-up diagnostic reports, then return to view comparative analytics."
            )
        else:
            # Top KPI Cards Strip
            st.markdown("### 1 · Key Clinical Health Indicators (Current Status)")
            kpi_cols = st.columns(5)

            # Calculate latest parameter values
            latest_params = {}
            for r in sorted(filtered_reports, key=lambda x: x.get("date", "")):
                latest_params[r["parameter"]] = r

            hba1c_latest = latest_params.get("HbA1c")
            fbg_latest = latest_params.get("Fasting Blood Glucose")
            sbp_latest = latest_params.get("Systolic Blood Pressure")
            chol_latest = latest_params.get("Total Cholesterol")
            creat_latest = latest_params.get("Serum Creatinine")

            with kpi_cols[0]:
                val = f"{hba1c_latest['value']}%" if hba1c_latest else "N/A"
                cls = "green" if (hba1c_latest and hba1c_latest["value"] <= 6.5) else "red"
                st.markdown(
                    f'<div class="kpi-card {cls}"><div class="kpi-val">{val}</div><div class="kpi-lbl">HbA1c (Glycemic)</div><div class="kpi-sub">Target: &lt; 6.5%</div></div>',
                    unsafe_allow_html=True,
                )

            with kpi_cols[1]:
                val = f"{fbg_latest['value']:.0f}" if fbg_latest else "N/A"
                cls = "green" if (fbg_latest and fbg_latest["value"] <= 110) else "amber"
                st.markdown(
                    f'<div class="kpi-card {cls}"><div class="kpi-val">{val}</div><div class="kpi-lbl">Fasting Glucose (mg/dL)</div><div class="kpi-sub">Target: 70 - 99</div></div>',
                    unsafe_allow_html=True,
                )

            with kpi_cols[2]:
                val = f"{sbp_latest['value']:.0f}" if sbp_latest else "N/A"
                cls = "green" if (sbp_latest and sbp_latest["value"] <= 125) else "amber"
                st.markdown(
                    f'<div class="kpi-card {cls}"><div class="kpi-val">{val}</div><div class="kpi-lbl">Systolic BP (mmHg)</div><div class="kpi-sub">Target: &lt; 120</div></div>',
                    unsafe_allow_html=True,
                )

            with kpi_cols[3]:
                val = f"{chol_latest['value']:.0f}" if chol_latest else "N/A"
                cls = "green" if (chol_latest and chol_latest["value"] <= 200) else "red"
                st.markdown(
                    f'<div class="kpi-card {cls}"><div class="kpi-val">{val}</div><div class="kpi-lbl">Total Cholesterol (mg/dL)</div><div class="kpi-sub">Target: &lt; 200</div></div>',
                    unsafe_allow_html=True,
                )

            with kpi_cols[4]:
                val = f"{creat_latest['value']:.2f}" if creat_latest else "N/A"
                cls = "green" if (creat_latest and creat_latest["value"] <= 1.2) else "amber"
                st.markdown(
                    f'<div class="kpi-card {cls}"><div class="kpi-val">{val}</div><div class="kpi-lbl">Serum Creatinine (mg/dL)</div><div class="kpi-sub">Target: 0.7 - 1.3</div></div>',
                    unsafe_allow_html=True,
                )

            st.write("")
            st.divider()

        # Section 2: Medication Impact Timeline & Comparison Grid (Prescription data REMOVED from Report Analysis per requirement)
        st.markdown("### 2 · Medication Impact Timeline & Regimen Shifts")

        col_tline, col_barchart = st.columns([1.2, 1])

        with col_tline:
            st.markdown('<div class="panel-card">', unsafe_allow_html=True)
            st.markdown("#### 📅 Chronological Medication Regimen Roadmap")
            if med_timeline:
                for period in med_timeline:
                    with st.expander(
                        f"📍 {period.get('period_name')} ({period.get('start_date')} to {period.get('end_date')})",
                        expanded=True,
                    ):
                        st.write(f"**Target:** {period.get('clinical_target', 'N/A')}")
                        st.write("**Medications in this interval:**")
                        for med_item in period.get("medications", []):
                            st.markdown(f"- 💊 `{med_item}`")
            else:
                st.caption("No medication timeline recorded. Upload reports in Reports Comparision tab.")
            st.markdown("</div>", unsafe_allow_html=True)

        with col_barchart:
            st.markdown('<div class="panel-card">', unsafe_allow_html=True)
            st.markdown("#### 📊 Parameter Comparison Across Medication Periods")

            if lab_reports:
                comp_summary = []
                target_biomarkers = selected_biomarkers or [
                    "HbA1c",
                    "Fasting Blood Glucose",
                    "Systolic Blood Pressure",
                    "Total Cholesterol",
                ]
                for p_name in target_biomarkers:
                    p_rows = [r for r in filtered_reports if r["parameter"] == p_name]
                    if len(p_rows) >= 2:
                        p_rows_sorted = sorted(p_rows, key=lambda x: x["date"])
                        comp_summary.append(
                            {
                                "Parameter": p_name,
                                "Baseline (Pre-Treatment)": p_rows_sorted[0]["value"],
                                "Current (Post-Treatment)": p_rows_sorted[-1]["value"],
                            }
                        )

                if comp_summary:
                    df_bar_comp = pd.DataFrame(comp_summary)
                    fig_bar = go.Figure()
                    fig_bar.add_trace(
                        go.Bar(
                            x=df_bar_comp["Parameter"],
                            y=df_bar_comp["Baseline (Pre-Treatment)"],
                            name="Baseline (Pre-Treatment)",
                            marker_color="#ef4444",
                        )
                    )
                    fig_bar.add_trace(
                        go.Bar(
                            x=df_bar_comp["Parameter"],
                            y=df_bar_comp["Current (Post-Treatment)"],
                            name="Current (Post-Treatment)",
                            marker_color="#22c55e",
                        )
                    )
                    fig_bar.update_layout(
                        barmode="group",
                        height=340,
                        margin={"l": 10, "r": 10, "t": 20, "b": 30},
                        legend={"orientation": "h", "y": -0.2},
                        plot_bgcolor="white",
                        paper_bgcolor="white",
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)
                else:
                    st.info("Select biomarkers with multiple visit reports to view before/after comparison bar chart.")
            else:
                st.caption("No laboratory data available for comparison chart.")
            st.markdown("</div>", unsafe_allow_html=True)

        st.divider()

        # Section 3: Condition Monitoring Table & Allergy Safety
        st.markdown("### 3 · Condition Monitoring & Allergy Safety Matrix")

        col_cond, col_all = st.columns([1.3, 1])

        with col_cond:
            st.markdown('<div class="panel-card">', unsafe_allow_html=True)
            st.markdown("#### 🩺 Condition Monitoring Trajectory Matrix")
            if lab_reports:
                cond_rows = []
                for p_name, p_def in impact_analyzer.PARAMETER_METRICS.items():
                    p_obs = [r for r in filtered_reports if r["parameter"] == p_name]
                    if p_obs:
                        p_obs_sorted = sorted(p_obs, key=lambda x: x["date"])
                        first_val = p_obs_sorted[0]["value"]
                        last_val = p_obs_sorted[-1]["value"]
                        delta = last_val - first_val

                        if abs(delta) < 0.1:
                            trend_icon = "➡️ Stable"
                        elif (delta < 0 and p_def.get("lower_is_better")) or (delta > 0 and not p_def.get("lower_is_better")):
                            trend_icon = "🟢 Favorable Trajectory"
                        else:
                            trend_icon = "🔴 Elevated"

                        cond_rows.append(
                            {
                                "Monitored Condition": p_def.get("condition"),
                                "Biomarker": p_name,
                                "Baseline": f"{first_val:.1f}",
                                "Current": f"{last_val:.1f} {p_def.get('unit')}",
                                "Reference Range": f"{p_def.get('normal_min')} - {p_def.get('normal_max')}",
                                "Observed Trajectory": trend_icon,
                            }
                        )

                if cond_rows:
                    st.dataframe(pd.DataFrame(cond_rows), use_container_width=True, hide_index=True)
                else:
                    st.info("No parameters match the active filter criteria.")
            else:
                st.caption("No laboratory records available for trajectory matrix.")
            st.markdown("</div>", unsafe_allow_html=True)

        with col_all:
            st.markdown('<div class="panel-card">', unsafe_allow_html=True)
            st.markdown("#### 🛡️ Patient Allergies & Clinical Warning Alerts")

            st.write("**Known Drug Allergies:**")
            allergies = current_patient.get("known_allergies", [])
            if allergies and allergies != ["None documented", "No known drug allergies"]:
                for a in allergies:
                    st.markdown(
                        f'<div class="alert-box critical">⚠️ <b>Allergy:</b> {a} (Immediate Hypersensitivity Risk)</div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.markdown(
                    '<div class="alert-box safe">✅ No known drug allergies documented.</div>', unsafe_allow_html=True
                )

            st.write("**Documented Adverse Drug Reactions:**")
            adv = current_patient.get("adverse_reactions", [])
            if adv and adv != ["None"]:
                for ad in adv:
                    st.markdown(
                        f'<div class="alert-box warning">🚫 <b>Adverse Reaction:</b> {ad}</div>', unsafe_allow_html=True
                    )
            else:
                st.caption("No adverse reactions recorded.")

            st.write("**Clinical Guidelines:**")
            st.info(
                "Clinicians should verify renal clearance (eGFR) prior to escalating biguanide or SGLT2 inhibitor doses."
            )
            st.markdown("</div>", unsafe_allow_html=True)

        st.divider()

        # Section 4: Predictive Health Analytics & Trajectory Forecasting
        st.markdown("### 4 · 🔮 Predictive Health Analytics & Biomarker Trajectory Forecasting")
        st.markdown(
            "Leverages longitudinal patient regression velocities across medication intervals to forecast future biomarker levels "
            "at 30, 60, and 90 days. Provides risk stratification, target normalcy projections, and what-if clinical simulations."
        )

        col_pred_sel, col_pred_info = st.columns([1.5, 2.5])
        with col_pred_sel:
            target_pred_param = st.selectbox(
                "Select Biomarker for Trajectory Forecasting:",
                options=[
                    "HbA1c",
                    "Fasting Blood Glucose",
                    "Systolic Blood Pressure",
                    "Total Cholesterol",
                    "Serum Creatinine",
                ],
                index=0,
                key="pred_param_selector",
            )
        with col_pred_info:
            p_metric_info = impact_analyzer.PARAMETER_METRICS.get(target_pred_param, {})
            st.caption(
                f"**Condition:** {p_metric_info.get('condition')} | **Target Range:** {p_metric_info.get('normal_min')} - {p_metric_info.get('normal_max')} {p_metric_info.get('unit')} | "
                f"**Clinical Direction:** {'Lower is better' if p_metric_info.get('lower_is_better') else 'Higher is better'}"
            )

        pred_res = impact_analyzer.predict_parameter_trajectory(filtered_reports, target_pred_param)

        if pred_res.get("status") == "success":
            f_pts = pred_res.get("forecast_points", [])
            curr_v = pred_res.get("current_value", 0.0)
            f_30 = f_pts[0]["forecast_value"] if len(f_pts) > 0 else curr_v
            f_60 = f_pts[1]["forecast_value"] if len(f_pts) > 1 else curr_v
            f_90 = f_pts[2]["forecast_value"] if len(f_pts) > 2 else curr_v

            kpi_p1, kpi_p2, kpi_p3, kpi_p4 = st.columns(4)
            unit = p_metric_info.get("unit", "")
            with kpi_p1:
                st.markdown(
                    f'<div class="kpi-card"><div class="kpi-val">{curr_v:.1f} {unit}</div><div class="kpi-lbl">Current Observed</div><div class="kpi-sub">Date: {pred_res.get("current_date")}</div></div>',
                    unsafe_allow_html=True,
                )
            with kpi_p2:
                st.markdown(
                    f'<div class="kpi-card green"><div class="kpi-val">{f_30:.1f} {unit}</div><div class="kpi-lbl">+30 Days Projected</div><div class="kpi-sub">Est: {f_pts[0]["forecast_date"] if f_pts else "—"}</div></div>',
                    unsafe_allow_html=True,
                )
            with kpi_p3:
                st.markdown(
                    f'<div class="kpi-card green"><div class="kpi-val">{f_60:.1f} {unit}</div><div class="kpi-lbl">+60 Days Projected</div><div class="kpi-sub">Est: {f_pts[1]["forecast_date"] if len(f_pts) > 1 else "—"}</div></div>',
                    unsafe_allow_html=True,
                )
            with kpi_p4:
                st.markdown(
                    f'<div class="kpi-card purple"><div class="kpi-val">{f_90:.1f} {unit}</div><div class="kpi-lbl">+90 Days Projected</div><div class="kpi-sub">Est: {f_pts[2]["forecast_date"] if len(f_pts) > 2 else "—"}</div></div>',
                    unsafe_allow_html=True,
                )

            st.write("")
            st.info(
                f"🎯 **Target Normalcy Projection:** {pred_res.get('target_achievement_estimate')}\n\n"
                f"⚡ **Monthly Velocity:** `{pred_res.get('monthly_velocity'):+.2f} {unit}/month` &nbsp;|&nbsp; "
                f"💊 **Therapeutic Response Efficacy:** `{pred_res.get('efficacy_rating')}`"
            )

            df_hist = impact_analyzer.get_time_series_data_by_parameter(filtered_reports, target_pred_param)
            if not df_hist.empty:
                fig_pred = go.Figure()

                n_min = p_metric_info.get("normal_min")
                n_max = p_metric_info.get("normal_max")
                if n_min is not None and n_max is not None:
                    fig_pred.add_hrect(
                        y0=n_min,
                        y1=n_max,
                        fillcolor="rgba(34, 197, 94, 0.12)",
                        line_width=0,
                        annotation_text=f"Normal Target ({n_min} - {n_max})",
                        annotation_position="bottom right",
                    )

                fig_pred.add_trace(
                    go.Scatter(
                        x=df_hist["date_str"],
                        y=df_hist["value"],
                        mode="lines+markers+text",
                        name="Observed Clinical History",
                        line={"color": "#0284c7", "width": 3},
                        marker={"size": 10, "color": "#0369a1"},
                        text=[f"{v:.1f}" for v in df_hist["value"]],
                        textposition="top center",
                    )
                )

                last_hist_date = df_hist["date_str"].iloc[-1]
                forecast_x = [last_hist_date] + [p["forecast_date"] for p in f_pts]
                forecast_y = [curr_v] + [p["forecast_value"] for p in f_pts]
                forecast_upper = [curr_v] + [p["upper_bound"] for p in f_pts]
                forecast_lower = [curr_v] + [p["lower_bound"] for p in f_pts]

                fig_pred.add_trace(
                    go.Scatter(
                        x=forecast_x + forecast_x[::-1],
                        y=forecast_upper + forecast_lower[::-1],
                        fill="toself",
                        fillcolor="rgba(124, 58, 237, 0.12)",
                        line={"color": "rgba(255,255,255,0)"},
                        name="Forecast Confidence Interval (95%)",
                        showlegend=True,
                        hoverinfo="skip",
                    )
                )

                fig_pred.add_trace(
                    go.Scatter(
                        x=forecast_x,
                        y=forecast_y,
                        mode="lines+markers+text",
                        name="Projected Trajectory (Under Active Regimen)",
                        line={"color": "#7c3aed", "width": 3, "dash": "dash"},
                        marker={"size": 10, "color": "#7c3aed", "symbol": "diamond"},
                        text=[f"{v:.1f} (Proj)" for v in forecast_y],
                        textposition="top center",
                    )
                )

                fig_pred.update_layout(
                    title=f"Predictive Trajectory Forecast: {target_pred_param} (Observed vs 90-Day Projection)",
                    xaxis_title="Date",
                    yaxis_title=f"{target_pred_param} ({unit})",
                    height=430,
                    margin={"l": 20, "r": 20, "t": 50, "b": 30},
                    plot_bgcolor="white",
                    paper_bgcolor="white",
                    font={"family": "Inter", "size": 12},
                )
                fig_pred.update_xaxes(showgrid=True, gridcolor="#f1f5f9")
                fig_pred.update_yaxes(showgrid=True, gridcolor="#f1f5f9")
                st.plotly_chart(fig_pred, use_container_width=True)

            with st.expander("🧪 Interactive What-If Medication Regimen Simulator", expanded=False):
                st.write(
                    f"Simulate potential biomarker trajectories for **{target_pred_param}** under alternate scenarios:"
                )
                sim_col1, sim_col2 = st.columns(2)
                with sim_col1:
                    sim_action = st.radio(
                        "Select Regimen Scenario to Simulate:",
                        [
                            "Scenario A: Continue Current Therapy (Maintenance)",
                            "Scenario B: Revert to Previous Medicine (De-escalation)",
                            "Scenario C: Regimen Intensification (Add SGLT2i / Second Agent)",
                        ],
                        index=0,
                    )
                with sim_col2:
                    if "Scenario A" in sim_action:
                        st.success(
                            f"**Projected 90-day {target_pred_param}:** `{f_90:.1f} {unit}`\n\n"
                            "Maintains observed downward trajectory towards target."
                        )
                    elif "Scenario B" in sim_action:
                        rebound_val = curr_v + (abs(curr_v - f_90) * 0.8)
                        st.warning(
                            f"**Projected 90-day {target_pred_param} (Rebound):** `{rebound_val:.1f} {unit}`\n\n"
                            "Risk of glycemic / clinical rebound due to withdrawal of second agent."
                        )
                    else:
                        intense_val = max(p_metric_info.get("normal_min", 4.0), f_90 * 0.92)
                        st.info(
                            f"**Projected 90-day {target_pred_param} (Intensified):** `{intense_val:.1f} {unit}`\n\n"
                            "Accelerated normalization; monitor renal markers and hypoglycemia risk."
                        )
        else:
            st.info("Insufficient laboratory records to compute trajectory forecast for this parameter.")

        st.divider()

        # Section 5: Export Power BI Datasets (CSV) - Prescription data ISOLATED in exports; Report Analysis = lab reports ONLY
        st.markdown("### 5 · Export Clean Datasets for Power BI Desktop")
        st.markdown(
            "All visual analytics above are live inside the application. Click below to export "
            "structured relational CSV datasets to load into Microsoft Power BI Desktop for offline modeling."
        )

        col_exp_btn, col_dl_pbi, col_dl_json = st.columns([1, 1, 1.2])
        with col_exp_btn:
            if st.button("📤 Export Relational CSVs to Disk (outputs/exports/)", type="primary", use_container_width=True):
                with st.spinner("Generating clean Power BI relational datasets..."):
                    exported_paths = orchestrator.export_longitudinal_powerbi_datasets(
                        patient_info=current_patient,
                        comparison_records=comparison_data,
                        prescriptions=[],
                        lab_reports=lab_reports,
                    )
                st.success("✅ Power BI CSV datasets successfully written to `outputs/exports/`!")
                st.toast("Power BI relational datasets exported!")
                for name, pth in exported_paths.items():
                    if os.path.exists(pth):
                        st.caption(f"📄 **{name.replace('_', ' ').title()}**: `{pth}` ({os.path.getsize(pth)} bytes)")

        with col_dl_pbi:
            if comparison_data:
                df_dl_comp = pd.DataFrame(comparison_data)
                csv_comp_bytes = df_dl_comp.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Download Medication Impact CSV",
                    data=csv_comp_bytes,
                    file_name=f"medication_impact_{selected_pid}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

        with col_dl_json:
            unified_json = orchestrator.generate_unified_clinical_json(
                patient_info=current_patient,
                comparison_records=comparison_data,
                prescriptions=[],
                lab_reports=lab_reports,
            )
            st.download_button(
                "⬇️ Download Unified Clinical JSON",
                data=json.dumps(unified_json, indent=2),
                file_name=f"clinical_intelligence_{selected_pid}_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json",
                use_container_width=True,
            )
