import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from io import BytesIO

# ==========================================
# ⚙️ إعدادات الصفحة
# ==========================================
st.set_page_config(page_title="المحلل الإحصائي الشامل (Pro)", layout="wide", page_icon="📊")

# ==========================================
# 🎨 الديكور والواجهة (CSS)
# ==========================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@300;400;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Tajawal', sans-serif; direction: rtl; background-color: #f8f9fa; }
    
    /* الهيدر */
    .hero-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 30px; border-radius: 15px; color: white; text-align: center;
        box-shadow: 0 10px 20px rgba(0,0,0,0.1); margin-bottom: 25px;
    }
    
    /* صندوق رأي المهندس */
    .engineer-insight {
        background-color: #e8f6f3; border-right: 6px solid #1abc9c;
        padding: 15px; border-radius: 8px; margin-top: 15px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05); color: #16a085;
    }
    .engineer-title { font-weight: bold; font-size: 1.1em; display: flex; align-items: center; gap: 8px; }
    
    /* صناديق الحالة */
    .success-box { background-color: #d1e7dd; color: #0f5132; padding: 15px; border-radius: 10px; border-right: 5px solid #198754; margin-bottom: 10px; }
    .warning-box { background-color: #fff3cd; color: #664d03; padding: 15px; border-radius: 10px; border-right: 5px solid #ffc107; margin-bottom: 10px; }
    .error-box { background-color: #f8d7da; color: #721c24; padding: 15px; border-radius: 10px; border-right: 5px solid #dc3545; margin-bottom: 10px; }

    /* البطاقات */
    .metric-card { background: white; padding: 15px; border-radius: 12px; border: 1px solid #e0e0e0; text-align: center; border-bottom: 4px solid #2a5298; }
    .metric-val { font-size: 1.8em; font-weight: bold; color: #2a5298; }
    
    /* الأزرار */
    .stButton>button { background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%); color: white; border: none; border-radius: 8px; font-weight: bold; width: 100%; }
    .stButton>button:hover { opacity: 0.9; color: white; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 🛠️ دوال المساعدة (Helpers & Caching)
# ==========================================
@st.cache_data(ttl=3600)
def load_data(file):
    try:
        return pd.read_csv(file) if file.name.endswith('.csv') else pd.read_excel(file, engine='openpyxl')
    except: return None

@st.cache_data
def convert_df_to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
    return output.getvalue()

def check_normality(data):
    if len(data) < 3: return True
    try:
        stat, p = stats.shapiro(data)
        return p > 0.05
    except: return True

def draw_card(title, value, icon="📊"):
    st.markdown(f"""<div class="metric-card"><div style="color:#666; font-size:0.9em;">{icon} {title}</div><div class="metric-val">{value}</div></div>""", unsafe_allow_html=True)

# --- دوال المهندس والاستشارة ---
def explain_hypothesis(p_value, test_name):
    if p_value < 0.05:
        return f"""<div class="engineer-insight"><div class="engineer-title">✅ النتيجة إيجابية (يوجد فرق حقيقي):</div><p>قيمة P-value أقل من 0.05. التأويل: العامل المدروس له تأثير حقيقي وليس صدفة.</p></div>"""
    else:
        return f"""<div class="engineer-insight" style="background-color:#fff3cd; border-color:#ffc107; color:#856404;"><div class="engineer-title">✋ النتيجة سلبية (لا يوجد فرق):</div><p>قيمة P-value أكبر من 0.05. التأويل: الفروقات بسيطة وتعتبر عشوائية.</p></div>"""

def explain_capability(cpk):
    if cpk < 1.0: return """<div class="engineer-insight" style="background-color:#f8d7da; border-color:#dc3545; color:#721c24;"><div class="engineer-title">🚨 العملية غير قادرة:</div><p>أنت تنتج الكثير من العيوب. أوقف الخط وافحص التباين.</p></div>"""
    elif cpk < 1.33: return """<div class="engineer-insight" style="background-color:#fff3cd; border-color:#ffc107; color:#856404;"><div class="engineer-title">⚠️ العملية مقبولة بحذر:</div><p>العملية تفي بالمواصفات بالكاد. راقبها جيداً.</p></div>"""
    else: return """<div class="engineer-insight"><div class="engineer-title">✅ العملية ممتازة:</div><p>العملية مستقرة وتقع داخل المواصفات بأمان.</p></div>"""

def check_data_health(data):
    report = []
    stat, p_norm = stats.shapiro(data)
    if p_norm > 0.05: report.append("<div class='success-box'>✅ التوزيع طبيعي (Normal).</div>")
    else: report.append("<div class='warning-box'>⚠️ التوزيع غير طبيعي.</div>")
    
    Q1, Q3 = data.quantile(0.25), data.quantile(0.75)
    IQR = Q3 - Q1
    outliers = data[((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR)))]
    if len(outliers) > 0: report.append(f"<div class='error-box'>🚨 يوجد {len(outliers)} قيم شاذة.</div>")
    else: report.append("<div class='success-box'>✅ لا توجد قيم شاذة.</div>")
    return "".join(report)

def analyze_variance_sources(df, target, factors):
    try:
        formula = f"{target} ~ " + " + ".join([f"C({f})" for f in factors])
        model = ols(formula, data=df).fit()
        aov = sm.stats.anova_lm(model, typ=2)
        total = aov['sum_sq'].sum()
        aov['Contribution_%'] = (aov['sum_sq'] / total) * 100
        return aov[['Contribution_%']].sort_values(by='Contribution_%', ascending=False)
    except: return None

# ==========================================
# 🚀 الهيكل الرئيسي (Main App)
# ==========================================

# 1. الهيدر
st.markdown("""
<div class="hero-header">
    <h1>🚀 المحلل الإحصائي الشامل</h1>
    <p>Data Science • Six Sigma • AI</p>
</div>
""", unsafe_allow_html=True)

# 2. القائمة الجانبية (مع الإدخال اليدوي الجديد)
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2804/2804702.png", width=80)
    st.title("لوحة التحكم")
    st.write("---")
    
    # اختيار المصدر
    data_source = st.radio("مصدر البيانات:", ("📂 رفع ملف (Excel/CSV)", "✍️ إدخال يدوي (جدول)", "🎲 بيانات تجريبية"))
    
    df = None # المتغير الرئيسي للبيانات
    
    if data_source == "📂 رفع ملف (Excel/CSV)":
        uploaded_file = st.file_uploader("اختر الملف:", type=['csv', 'xlsx'])
        if uploaded_file:
            df = load_data(uploaded_file)
            if df is None: st.error("❌ ملف تالف.")

    elif data_source == "✍️ إدخال يدوي (جدول)":
        st.info("عرف الأعمدة ثم املأ البيانات.")
        cols_in = st.text_input("أسماء الأعمدة (فواصل):", "الماكينة, الوزن, الطول")
        cols = [c.strip() for c in cols_in.split(',')]
        
        if 'manual_data' not in st.session_state:
            st.session_state.manual_data = pd.DataFrame([[""]*len(cols)]*5, columns=cols)
        
        if list(st.session_state.manual_data.columns) != cols:
             st.session_state.manual_data = pd.DataFrame([[""]*len(cols)]*5, columns=cols)

        st.write("▼ البيانات:")
        edited_df = st.data_editor(st.session_state.manual_data, num_rows="dynamic", use_container_width=True)
        
        if not edited_df.empty:
            for c in edited_df.columns: edited_df[c] = pd.to_numeric(edited_df[c], errors='ignore')
            df = edited_df.dropna(how='all')
            st.session_state.manual_data = edited_df

    elif data_source == "🎲 بيانات تجريبية":
        if st.button("توليد بيانات عشوائية"):
            np.random.seed(42)
            d = {'الإنتاجية': np.random.normal(100, 15, 100), 'الماكينة': np.random.choice(['A', 'B'], 100), 'عيوب': np.random.poisson(2, 100)}
            d['الإنتاجية'] += np.where(d['الماكينة']=='A', 10, 0)
            df = pd.DataFrame(d)
            st.success("تم التوليد!")

# 3. جسم التطبيق والتحليلات
if df is not None and not df.empty:
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    tabs = st.tabs(["📋 استكشاف", "🔎 المصدر والصحة", "⚖️ فروقات", "🏭 جودة", "🧪 تصميم", "🤖 ذكاء", "📏 عينات"])

    # --- Tab 1: Explore ---
    with tabs[0]:
        st.subheader("📊 نظرة عامة")
        c1, c2, c3 = st.columns(3)
        with c1: draw_card("عدد السجلات", df.shape[0])
        with c2: draw_card("عدد الأعمدة", df.shape[1])
        with c3: draw_card("رقمية", len(num_cols))
        
        st.markdown("---")
        cm, cd = st.columns([2, 1])
        with cm:
            if num_cols:
                tc = st.selectbox("رسم التوزيع:", num_cols)
                fig = px.histogram(df, x=tc, marginal="box", template="plotly_white", color_discrete_sequence=['#2a5298'])
                st.plotly_chart(fig, use_container_width=True)
        with cd:
            if num_cols: st.dataframe(df[tc].describe(), use_container_width=True)

        st.download_button("📥 تحميل البيانات (Excel)", convert_df_to_excel(df), "data.xlsx")

    # --- Tab 2: Health & Source ---
    with tabs[1]:
        st.subheader("🔎 الصحة والمصدر")
        ch, cs = st.columns(2)
        with ch:
            st.markdown("##### 1️⃣ صحة البيانات")
            if num_cols:
                cc = st.selectbox("فحص عمود:", num_cols, key="hc")
                st.markdown(check_data_health(df[cc].dropna()), unsafe_allow_html=True)
        with cs:
            st.markdown("##### 2️⃣ مصدر التشتت")
            if num_cols and cat_cols:
                sy = st.selectbox("النتيجة:", num_cols, key="sy")
                sx = st.multiselect("العوامل:", cat_cols, key="sx")
                if st.button("تحليل") and sx:
                    res = analyze_variance_sources(df, sy, sx)
                    if res is not None:
                        fig = px.pie(res, values='Contribution_%', names=res.index, title="نسبة المساهمة")
                        st.plotly_chart(fig, use_container_width=True)
                        top = res.index[0]
                        st.markdown(f"<div class='engineer-insight'>العامل الأقوى تأثيراً هو <b>{top}</b></div>", unsafe_allow_html=True)

    # --- Tab 3: Hypothesis ---
    with tabs[2]:
        st.subheader("⚖️ اختبار الفروقات")
        if num_cols and cat_cols:
            c1, c2 = st.columns(2)
            y = c1.selectbox("النتيجة (Y):", num_cols, key="hy")
            x = c2.selectbox("المجموعة (X):", cat_cols, key="hx")
            if st.button("تحليل الفروقات"):
                grps = df[x].unique()
                if len(grps) >= 2:
                    dat = [df[df[x]==g][y] for g in grps]
                    if len(grps)==2: s, p = stats.ttest_ind(dat[0], dat[1]); tn="T-Test"
                    else: s, p = stats.f_oneway(*dat); tn="ANOVA"
                    
                    st.plotly_chart(px.box(df, x=x, y=y, color=x, template="plotly_white"), use_container_width=True)
                    st.markdown(explain_hypothesis(p, tn), unsafe_allow_html=True)

    # --- Tab 4: Quality ---
    with tabs[3]:
        st.subheader("🏭 ضبط الجودة")
        if num_cols:
            qc = st.selectbox("المتغير:", num_cols, key="qc")
            mu, sigma = df[qc].mean(), df[qc].std()
            usl = st.number_input("USL:", value=mu+4*sigma)
            lsl = st.number_input("LSL:", value=mu-4*sigma)
            
            c_ctrl, c_cap = st.columns(2)
            with c_ctrl:
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=df[qc], mode='lines+markers'))
                fig.add_hline(y=mu+3*sigma, line_color='red', line_dash='dash')
                fig.add_hline(y=mu-3*sigma, line_color='red', line_dash='dash')
                fig.add_hline(y=mu, line_color='green')
                fig.update_layout(template="plotly_white", title="Control Chart")
                st.plotly_chart(fig, use_container_width=True)
            with c_cap:
                Cp = (usl - lsl) / (6 * sigma)
                Cpk = min((usl - mu)/(3*sigma), (mu - lsl)/(3*sigma))
                st.metric("Cpk", f"{Cpk:.2f}")
                st.markdown(explain_capability(Cpk), unsafe_allow_html=True)

    # --- Tab 5: DOE ---
    with tabs[4]:
        st.subheader("🧪 تحليل التفاعل")
        if len(num_cols)>0 and (len(num_cols)+len(cat_cols))>=3:
            dy = st.selectbox("Y:", num_cols, key="dy")
            dx1 = st.selectbox("X1:", [c for c in df.columns if c!=dy], key="dx1")
            dx2 = st.selectbox("X2:", [c for c in df.columns if c!=dy and c!=dx1], key="dx2")
            if st.button("رسم التفاعل"):
                d_doe = df.groupby([dx1, dx2])[dy].mean().reset_index()
                st.plotly_chart(px.line(d_doe, x=dx1, y=dy, color=dx2, markers=True), use_container_width=True)

    # --- Tab 6: AutoML ---
    with tabs[5]:
        st.subheader("🤖 الذكاء الاصطناعي")
        mt = st.radio("النوع:", ["كشف المؤثرات", "صائد الشواذ", "تجميع"], horizontal=True)
        if mt == "كشف المؤثرات":
            tm = st.selectbox("الهدف:", num_cols, key="tm")
            fm = st.multiselect("المؤثرات:", [c for c in num_cols if c!=tm], key="fm")
            if st.button("تحليل") and fm:
                rf = RandomForestRegressor(100).fit(df[fm].fillna(0), df[tm].fillna(0))
                imp = pd.DataFrame({'F': fm, 'I': rf.feature_importances_}).sort_values('I', ascending=False)
                st.plotly_chart(px.bar(imp, x='I', y='F', orientation='h'), use_container_width=True)
        elif mt == "صائد الشواذ":
            sc = st.selectbox("العمود:", num_cols, key="sc")
            if st.button("كشف"):
                iso = IsolationForest(contamination=0.05).fit(df[[sc]].fillna(0))
                df['Iso'] = iso.predict(df[[sc]].fillna(0))
                st.plotly_chart(px.scatter(df, y=sc, color=df['Iso'].astype(str)), use_container_width=True)
        elif mt == "تجميع":
            kc = st.multiselect("أعمدة:", num_cols, key="kc")
            k = st.slider("مجموعات:", 2, 8, 3)
            if st.button("تجميع") and len(kc)>=2:
                km = KMeans(k).fit(StandardScaler().fit_transform(df[kc].dropna()))
                df['Clst'] = km.labels_.astype(str)
                st.plotly_chart(px.scatter_matrix(df, dimensions=kc, color='Clst'), use_container_width=True)

    # --- Tab 7: Planner ---
    with tabs[6]:
        st.subheader("📏 حجم العينة")
        c1, c2 = st.columns(2)
        cl = c1.selectbox("الثقة:", [0.90, 0.95, 0.99], index=1)
        me = c2.number_input("الخطأ (%):", 1.0, 10.0, 5.0)/100
        z = {0.90:1.645, 0.95:1.96, 0.99:2.576}[cl]
        n = (z**2 * 0.25)/(me**2)
        st.markdown(f"<div class='success-box' style='text-align:center'><h1>{int(n)+1}</h1></div>", unsafe_allow_html=True)

else:
    st.info("👈 يرجى اختيار مصدر بيانات من القائمة الجانبية.")
