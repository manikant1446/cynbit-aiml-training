import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import io, warnings
warnings.filterwarnings('ignore')

# ── PDF import (optional) ─────────────────────────────────────────────────────
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from reportlab.lib.units import cm
    PDF_OK = True
except ImportError:
    PDF_OK = False

st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓", layout="centered")

st.markdown("""
<style>
.header{background:linear-gradient(135deg,#1a3a4a,#0d6e8a);color:white;padding:1.4rem 2rem;border-radius:12px;margin-bottom:1.5rem}
.header h1{margin:0;font-size:1.7rem}
.header p{margin:.3rem 0 0;opacity:.8;font-size:.85rem}
.step{background:#f8fafc;border-left:4px solid #0d6e8a;border-radius:8px;padding:.7rem 1.2rem;margin:1.2rem 0 .8rem}
.step h4{margin:0;color:#0d6e8a;font-size:1rem}
.kpi{text-align:center;background:white;border:1px solid #e2e8f0;border-radius:10px;padding:.9rem}
.kpi .val{font-size:1.7rem;font-weight:700;color:#0d6e8a}
.kpi .lbl{font-size:.72rem;color:#64748b;text-transform:uppercase;letter-spacing:.05em}
.result-box{border-radius:12px;padding:1.4rem 1.8rem;margin-top:1rem;border:2px solid}
.stButton>button{background:linear-gradient(135deg,#0d6e8a,#14b8a6);color:white;border:none;border-radius:8px;font-weight:600;width:100%;padding:.6rem;font-size:1rem}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="header">
  <h1>🎓 Student Performance Predictor</h1>
  <p>Upload dataset → Analyze attendance → Predict performance → Visualize → Report</p>
</div>
""", unsafe_allow_html=True)

FEATURES = ["attendance_percent","study_hours_per_day","assignments_completed","prev_exam_score","sleep_hours"]
TARGET   = "performance_category"
COLORS   = {"High":"#16a34a","Medium":"#ca8a04","Low":"#dc2626"}
BG       = {"High":"#dcfce7","Medium":"#fef9c3","Low":"#fee2e2"}

SAMPLE_CSV = """student_id,name,attendance_percent,study_hours_per_day,assignments_completed,prev_exam_score,sleep_hours,performance_category
1,Aarav Sharma,92,4.5,18,78,7,High
2,Priya Patel,85,3.2,15,65,6,Medium
3,Rahul Verma,60,1.5,8,45,5,Low
4,Anjali Singh,95,5.0,20,88,8,High
5,Vikram Gupta,72,2.8,12,58,6,Medium
6,Neha Joshi,88,4.0,17,74,7,High
7,Arjun Kumar,45,1.0,5,35,5,Low
8,Sunita Rao,78,3.5,14,62,7,Medium
9,Karan Mehta,91,4.8,19,82,8,High
10,Pooja Nair,55,1.8,7,40,4,Low
11,Rohit Sharma,83,3.0,13,60,6,Medium
12,Divya Iyer,97,5.5,20,92,8,High
13,Amit Tiwari,63,2.0,9,48,5,Low
14,Sneha Gupta,80,3.3,15,66,7,Medium
15,Nikhil Jain,90,4.2,18,80,7,High
16,Kavya Reddy,87,4.1,17,76,8,High
17,Siddharth Pal,50,1.2,6,38,5,Low
18,Meena Choudhary,76,2.9,13,57,6,Medium
19,Akash Pandey,93,4.7,19,85,8,High
20,Riya Desai,68,2.3,10,52,5,Low
21,Harshit Agarwal,84,3.6,16,68,7,Medium
22,Tanvi Mishra,96,5.2,20,90,8,High
23,Shubham Yadav,58,1.6,8,42,5,Low
24,Nisha Bhatt,81,3.1,14,63,6,Medium
25,Gaurav Saxena,89,4.3,18,79,7,High
26,Pallavi Shah,74,2.7,11,55,6,Medium
27,Deepak Sinha,48,1.1,5,36,4,Low
28,Ananya Kapoor,94,4.9,19,86,8,High
29,Varun Bansal,66,2.1,10,50,5,Low
30,Kritika Dixit,79,3.2,14,64,7,Medium
31,Manish Tripathi,86,3.9,17,73,7,High
32,Swati Kulkarni,61,1.9,9,46,5,Low
33,Ashish Srivastava,75,2.8,12,58,6,Medium
34,Shruti Bajaj,92,4.6,19,83,8,High
35,Piyush Malhotra,53,1.4,7,39,5,Low
36,Aditi Rastogi,88,4.0,17,75,7,High
37,Rajesh Kumar,70,2.5,11,53,6,Medium
38,Lakshmi Venkat,95,5.1,20,89,8,High
39,Suresh Patil,57,1.7,8,43,5,Low
40,Geeta Pillai,82,3.4,15,67,7,Medium
41,Aditya Chauhan,91,4.5,18,81,8,High
42,Mamta Goyal,64,2.0,9,49,5,Low
43,Vivek Dubey,77,3.0,13,59,6,Medium
44,Preethi Krishnan,93,4.8,19,84,8,High
45,Sanjay Rawat,46,1.0,5,34,4,Low
46,Usha Menon,85,3.7,16,70,7,Medium
47,Tarun Bose,90,4.4,18,80,7,High
48,Nalini Aggarwal,59,1.8,8,44,5,Low
49,Chirag Lal,78,3.1,14,62,6,Medium
50,Simran Kohli,96,5.3,20,91,8,High"""

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Upload
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="step"><h4>📂 Step 1 — Upload Student Dataset</h4></div>', unsafe_allow_html=True)
uploaded = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")

if not uploaded:
    st.info("Upload a CSV with columns: `attendance_percent`, `study_hours_per_day`, `assignments_completed`, `prev_exam_score`, `sleep_hours`, `performance_category`")
    st.download_button("⬇️ Download Sample CSV", SAMPLE_CSV, "sample_students.csv", "text/csv")
    st.stop()

df = pd.read_csv(uploaded)
missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
if missing:
    st.error(f"Missing columns: {missing}")
    st.stop()
df = df.dropna(subset=FEATURES + [TARGET])
st.success(f"✅ {len(df)} students loaded")

# Train model once
le = LabelEncoder()
y  = le.fit_transform(df[TARGET])
X  = df[FEATURES]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_tr, y_tr)
acc = clf.score(X_te, y_te)
df["predicted"] = le.inverse_transform(clf.predict(X))

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Attendance Analysis
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="step"><h4>📅 Step 2 — Attendance Analysis</h4></div>', unsafe_allow_html=True)

avg_att  = df["attendance_percent"].mean()
low_att  = (df["attendance_percent"] < 75).sum()
high_att = (df["attendance_percent"] >= 90).sum()

c1,c2,c3 = st.columns(3)
with c1: st.markdown(f'<div class="kpi"><div class="val">{avg_att:.1f}%</div><div class="lbl">Avg Attendance</div></div>', unsafe_allow_html=True)
with c2: st.markdown(f'<div class="kpi"><div class="val" style="color:#dc2626">{low_att}</div><div class="lbl">Below 75% (At Risk)</div></div>', unsafe_allow_html=True)
with c3: st.markdown(f'<div class="kpi"><div class="val" style="color:#16a34a">{high_att}</div><div class="lbl">Above 90% (Excellent)</div></div>', unsafe_allow_html=True)

fig_att, ax = plt.subplots(figsize=(6,2.8))
att_by_cat = df.groupby(TARGET)["attendance_percent"].mean().reindex(["High","Medium","Low"])
bars = ax.bar(att_by_cat.index, att_by_cat.values,
              color=[COLORS.get(c,"#888") for c in att_by_cat.index],
              width=0.4, edgecolor='white', linewidth=1.5, zorder=3)
for bar,val in zip(bars, att_by_cat.values):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
            f"{val:.1f}%", ha='center', fontweight='600', fontsize=10)
ax.set_ylabel("Avg Attendance %"); ax.set_ylim(0,110)
ax.grid(axis='y', alpha=0.3, zorder=0); ax.set_axisbelow(True)
fig_att.tight_layout(); st.pyplot(fig_att); plt.close(fig_att)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Predict Individual Student
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="step"><h4>🔮 Step 3 — Predict Student Performance</h4></div>', unsafe_allow_html=True)
st.markdown("Student ki details bharein aur **Predict** dabayein:")

col1, col2, col3 = st.columns(3)
with col1:
    s_name   = st.text_input("Student Name", "")
    s_attend = st.number_input("Attendance %", 0, 100, 75)
with col2:
    s_prev   = st.number_input("Previous Exam Score (0-100)", 0, 100, 60)
    s_study  = st.number_input("Study Hours / Day", 0.0, 12.0, 3.0, 0.5)
with col3:
    s_assign = st.number_input("Assignments Completed (0-20)", 0, 20, 12)
    s_sleep  = st.number_input("Sleep Hours / Day", 0, 12, 7)

if st.button("🔮 Predict Performance"):
    inp   = np.array([[s_attend, s_study, s_assign, s_prev, s_sleep]])
    pred  = le.inverse_transform(clf.predict(inp))[0]
    proba = clf.predict_proba(inp)[0]

    emoji = {"High":"🌟","Medium":"📘","Low":"⚠️"}.get(pred,"")
    tips  = []
    if s_attend < 75:  tips.append("📅 Attendance 75% se upar lao")
    if s_prev   < 50:  tips.append("📊 Weak subjects pe focus karo")
    if s_study  < 3:   tips.append("📚 Roz 3-4 ghante padho")
    if s_sleep  < 6:   tips.append("😴 7-8 ghante neend lo")
    if s_assign < 10:  tips.append("📝 Zyada assignments complete karo")
    if not tips:       tips.append("✅ Bahut achha! Isi tarah mehnat karte raho.")

    name_str = f" — {s_name}" if s_name.strip() else ""
    st.markdown(f"""
    <div class="result-box" style="background:{BG[pred]};border-color:{COLORS[pred]}">
      <p style="margin:0;font-size:.8rem;color:#475569;text-transform:uppercase;letter-spacing:.05em">Prediction Result{name_str}</p>
      <h2 style="margin:.3rem 0;color:{COLORS[pred]}">{emoji} {pred} Performer</h2>
      <hr style="border:1px solid {COLORS[pred]}33;margin:.6rem 0">
      <p style="margin:0;font-weight:600;color:#1e293b">Confidence Scores:</p>
    """, unsafe_allow_html=True)

    for cat, prob in zip(le.classes_, proba):
        pct = prob * 100
        st.markdown(f"""
        <div style="margin:.3rem 0">
          <div style="display:flex;justify-content:space-between">
            <span style="font-size:.9rem">{cat}</span>
            <span style="font-weight:700;color:{COLORS[cat]}">{pct:.1f}%</span>
          </div>
          <div style="background:#e2e8f0;border-radius:999px;height:8px">
            <div style="width:{pct}%;background:{COLORS[cat]};height:8px;border-radius:999px"></div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    tips_html = "".join(f"<p style='margin:.25rem 0'>• {t}</p>" for t in tips)
    st.markdown(f"""
      <p style="margin:.8rem 0 .3rem;font-weight:600;color:#1e293b">💡 Suggestions:</p>
      <div style="font-size:.88rem;color:#475569">{tips_html}</div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Visualize Results
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="step"><h4>📊 Step 4 — Visualize Results</h4></div>', unsafe_allow_html=True)

cl, cr = st.columns(2)
with cl:
    fig1, axes = plt.subplots(1,2,figsize=(6,3.2))
    for ax,col,title in zip(axes,[TARGET,"predicted"],["Actual","Predicted"]):
        counts = df[col].value_counts().reindex(["High","Medium","Low"])
        ax.bar(counts.index, counts.values,
               color=[COLORS.get(c,"#888") for c in counts.index],
               edgecolor='white', linewidth=1.2, width=0.5, zorder=3)
        for i,(idx,v) in enumerate(counts.items()):
            ax.text(i, v+0.15, str(v), ha='center', fontsize=9, fontweight='600')
        ax.set_title(title, fontweight='700', fontsize=10)
        ax.grid(axis='y', alpha=0.3, zorder=0); ax.set_axisbelow(True)
    fig1.tight_layout(); st.pyplot(fig1); plt.close(fig1)

with cr:
    fig2, ax2 = plt.subplots(figsize=(5,3.2))
    for cat,color in COLORS.items():
        sub = df[df[TARGET]==cat]
        ax2.scatter(sub["attendance_percent"], sub["prev_exam_score"],
                    c=color, label=cat, s=55, alpha=0.8, edgecolors='white', linewidths=0.6)
    ax2.set_xlabel("Attendance %", fontsize=9)
    ax2.set_ylabel("Exam Score", fontsize=9)
    ax2.set_title("Attendance vs Score", fontweight='700', fontsize=10)
    ax2.legend(fontsize=8); ax2.grid(alpha=0.2)
    fig2.tight_layout(); st.pyplot(fig2); plt.close(fig2)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Summary Report (PDF)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="step"><h4>📄 Step 5 — Generate Summary Report</h4></div>', unsafe_allow_html=True)

if not PDF_OK:
    st.warning("⚠️ reportlab not installed. Run: `pip3 install reportlab` then restart the app.")
else:
    if st.button("📥 Generate & Download PDF Report"):
        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=A4,
                                leftMargin=2*cm, rightMargin=2*cm,
                                topMargin=2*cm, bottomMargin=2*cm)
        styles = getSampleStyleSheet()
        story  = []

        TS = ParagraphStyle('T', parent=styles['Title'], fontSize=20,
                            textColor=colors.HexColor('#0d6e8a'), spaceAfter=6)
        SH = ParagraphStyle('SH', parent=styles['Heading2'], fontSize=13,
                            textColor=colors.HexColor('#0d6e8a'), spaceAfter=4)

        story.append(Paragraph("Student Performance Report", TS))
        story.append(Paragraph(f"Total Students: {len(df)}  |  Model Accuracy: {acc*100:.1f}%", styles['Normal']))
        story.append(Spacer(1, 0.5*cm))

        # Attendance table
        story.append(Paragraph("1. Attendance Analysis", SH))
        t1 = Table([["Metric","Value"],
                    ["Average Attendance", f"{avg_att:.1f}%"],
                    ["At Risk (< 75%)", str(low_att)],
                    ["Excellent (>= 90%)", str(high_att)]],
                   colWidths=[9*cm,6*cm])
        t1.setStyle(TableStyle([
            ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#0d6e8a')),
            ('TEXTCOLOR', (0,0),(-1,0),colors.white),
            ('FONTNAME',  (0,0),(-1,0),'Helvetica-Bold'),
            ('FONTSIZE',  (0,0),(-1,-1),10),
            ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#f0f9ff'),colors.white]),
            ('GRID',(0,0),(-1,-1),0.5,colors.HexColor('#cbd5e1')),
            ('PADDING',(0,0),(-1,-1),6),
        ]))
        story.append(t1); story.append(Spacer(1,0.3*cm))

        # Attendance chart
        fa,aa = plt.subplots(figsize=(5,2.8))
        aa.bar(att_by_cat.index, att_by_cat.values,
               color=[COLORS.get(c,"#888") for c in att_by_cat.index],
               width=0.4, edgecolor='white', zorder=3)
        aa.set_ylabel("Avg Attendance %"); aa.set_ylim(0,110)
        aa.grid(axis='y',alpha=0.3,zorder=0); aa.set_axisbelow(True)
        fa.tight_layout()
        ib1 = io.BytesIO(); fa.savefig(ib1, format='png', dpi=120, bbox_inches='tight')
        ib1.seek(0); plt.close(fa)
        story.append(RLImage(ib1, width=11*cm, height=6*cm))
        story.append(Spacer(1,0.4*cm))

        # Prediction summary
        story.append(Paragraph("2. Prediction Summary", SH))
        pred_counts = df["predicted"].value_counts()
        t2 = Table([["Category","Count","% of Total"]] +
                   [[cat, str(pred_counts.get(cat,0)), f"{pred_counts.get(cat,0)/len(df)*100:.1f}%"]
                    for cat in ["High","Medium","Low"]],
                   colWidths=[6*cm,4.5*cm,4.5*cm])
        t2.setStyle(TableStyle([
            ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#0d6e8a')),
            ('TEXTCOLOR', (0,0),(-1,0),colors.white),
            ('FONTNAME',  (0,0),(-1,0),'Helvetica-Bold'),
            ('FONTSIZE',  (0,0),(-1,-1),10),
            ('BACKGROUND',(0,1),(-1,1),colors.HexColor('#dcfce7')),
            ('BACKGROUND',(0,2),(-1,2),colors.HexColor('#fef9c3')),
            ('BACKGROUND',(0,3),(-1,3),colors.HexColor('#fee2e2')),
            ('GRID',(0,0),(-1,-1),0.5,colors.HexColor('#cbd5e1')),
            ('PADDING',(0,0),(-1,-1),6),
        ]))
        story.append(t2); story.append(Spacer(1,0.4*cm))

        # Student-wise table
        story.append(Paragraph("3. Student-wise Details (first 30)", SH))
        name_col = "name" if "name" in df.columns else None
        hdr  = (["Name"] if name_col else []) + ["Attend %","Prev Score","Actual","Predicted"]
        rows = [hdr]
        for _,row in df.head(30).iterrows():
            r = []
            if name_col: r.append(str(row[name_col])[:18])
            r += [f"{row['attendance_percent']:.0f}%",
                  f"{row['prev_exam_score']:.0f}",
                  row[TARGET], row["predicted"]]
            rows.append(r)
        cw = ([4.5*cm] if name_col else []) + [2.8*cm,2.8*cm,3*cm,3*cm]
        t3 = Table(rows, colWidths=cw)
        t3.setStyle(TableStyle([
            ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#0d6e8a')),
            ('TEXTCOLOR', (0,0),(-1,0),colors.white),
            ('FONTNAME',  (0,0),(-1,0),'Helvetica-Bold'),
            ('FONTSIZE',  (0,0),(-1,-1),8),
            ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#f8fafc'),colors.white]),
            ('GRID',(0,0),(-1,-1),0.3,colors.HexColor('#cbd5e1')),
            ('PADDING',(0,0),(-1,-1),4),
        ]))
        story.append(t3)
        story.append(Spacer(1,0.5*cm))
        story.append(Paragraph("Generated by Student Performance Predictor · Random Forest Classifier",
                               ParagraphStyle('ft',parent=styles['Normal'],fontSize=8,
                                              textColor=colors.HexColor('#94a3b8'))))
        doc.build(story)
        buf.seek(0)
        st.download_button("⬇️ Download PDF Report", buf, "student_report.pdf", "application/pdf")
        st.success("✅ PDF ready!")
