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
# ⚙️ إعدادات الصفحة (Page Config)
# ==========================================
st.set_page_config(
    page_title="المحلل الإحصائي الشامل (Pro)", 
    layout="wide", 
    page_icon="🧠",
    initial_sidebar_state="expanded"
)

# ==========================================
# 🎨 الديكور والتصميم (CSS / UI)
# ==========================================
st.markdown("""
<style>
    /* استيراد خط تجوال العربي */
    @import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@300;400;700;800&display=swap');

    /* 1. إعدادات الجسم العامة */
    html, body, [class*="css"] {
        font-family: 'Tajawal', sans-serif;
        direction: rtl;
        background-color: #f8f9fa;
    }

    /* 2. القائمة الجانبية */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e3c72 0%, #2a5298 100%);
        color: white;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] p, [data-testid="stSidebar"] label {
        color: #fff !important;
    }

    /* 3. الهيدر الرئيسي (Hero) */
    .hero-header {
        background: linear-gradient(135deg, #2a5298 0%, #1e3c72 100%);
        padding: 30px;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 20px rgba(0,0,0,0.1);
        margin-bottom: 25px;
        border: 1px solid rgba(255,255,255,0.2);
    }
    .hero-header h1 { margin: 0; font-size: 2.2em; font-weight: 800; }
    .hero-header p { opacity: 0.9; font-size: 1.1em; margin-top: 10px; }

    /* 4. البطاقات (Cards) */
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border-right: 5px solid #2a5298;
        transition: transform 0.2s;
        text-align: center;
    }
    .metric-card:hover { transform: translateY(-3px); box-shadow: 0 8px 15px rgba(0,0,0,0.1); }
    .metric-title { color: #6c757d; font-size: 0.9em; font-weight: bold; margin-bottom: 5px; }
    .metric-value { color: #1e3c72; font-size: 1.8em; font-weight: 800; }

    /* 5. صناديق التنبيه ورأي المهندس */
    .engineer-insight {
        background-color: #e8f6f3; border-right: 6px solid #1abc9c;
        padding: 15px; border-radius: 8px; margin-top: 15px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05); color: #16a085; text-align: right;
    }
    .engineer-title { font-weight: bold; font-size: 1.1em; margin-bottom: 8px; display: flex; align-items: center; gap: 8px; }
    
    .success-box { background-color: #d1e7dd; color: #0f5132; padding: 15px; border-radius: 10px; border-right: 5px solid #198754; margin-bottom: 10px; }
    .warning-box { background-color: #fff3cd; color: #664d03; padding: 15px; border-radius: 10px; border-right: 5px solid #ffc107; margin-bottom: 10px; }
    .error-box { background-color: #f8d7da; color: #842029; padding: 15px; border-radius: 10px; border-right: 5px solid #dc3545; margin-bottom: 10px; }

    /* 6. تحسين الأزرار */
    .stButton>button {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        color: white; border: none; border-radius: 8px; font-weight: bold; width: 100%; padding: 10px;
    }
    .stButton>button:hover { opacity: 0.9; color: white; transform: scale(1.01); }

</style>
""", unsafe_allow_html=True)

# ==========================================
# 🛠️ دوال المساعدة (Helpers)
# ==========================================

def draw_card(title, value, icon="📊"):
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">{icon} {title}</div>
        <div class="metric-value">{value}</div>
    </div>
    """, unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def load_data(file):
    try:
        if file.name.endswith('.csv'):
            return pd.read_csv(file)
        else:
            return pd.read_excel(file, engine='openpyxl')
    except Exception:
        return None

@st.cache_data
def convert_df_to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
    return output.getvalue()

def check_normality(data):
    if len(data) < 3: return True
    try:
        stat, p_value = stats.shapiro(data)
        return p_value > 0.05
    except: return True

# --- دوال التحليل الهندسي ---

def explain_hypothesis(p_value, test_name):
    """تفسير نتائج الفروقات"""
    if p_value < 0.05:
        return f"""
        <div class="engineer-insight">
            <div class="engineer-title">💡 رأي المهندس المختص:</div>
            <p><b>✅ النتيجة إيجابية (يوجد فرق حقيقي):</b></p>
            <ul>
                <li>قيمة P-value أقل من 0.05، مما يعني أن الاختلاف الذي تراه <b>ليس صدفة</b>.</li>
                <li><b>التأويل:</b> العامل الذي تدرسه له تأثير حقيقي ومؤثر على النتيجة.</li>
                <li><b>التوصية:</b> يمكنك اعتماد هذا التغيير أو النتيجة بثقة إحصائية 95%.</li>
            </ul>
        </div>
        """
    else:
        return f"""
        <div class="engineer-insight" style="background-color: #fdf2e9; border-color: #e67e22; color: #d35400;">
            <div class="engineer-title">💡 رأي المهندس المختص:</div>
            <p><b>✋ النتيجة سلبية (لا يوجد فرق):</b></p>
            <ul>
                <li>قيمة P-value أكبر من 0.05.</li>
                <li><b>التأويل:</b> الفروقات التي تراها بسيطة جداً وتعتبر "ضجيجاً" (Noise) أو صدفة.</li>
                <li><b>التوصية:</b> لا تتخذ قراراً مكلفاً بناءً على هذه البيانات، المجموعات متساوية.</li>
            </ul>
        </div>
        """

def explain_capability(cpk):
    """تفسير الجودة Cpk"""
    if cpk < 1.0:
        return """
        <div class="engineer-insight" style="background-color: #fadbd8; border-color: #e74c3c; color: #c0392b;">
            <div class="engineer-title">🚨 تحذير هندسي عاجل:</div>
            <p><b>العملية غير قادرة (Not Capable):</b></p>
            <p>أنت تنتج كميات كبيرة من المنتجات المعيبة (Scrap). <b>الإجراء:</b> أوقف الإنتاج وافحص أسباب التباين فوراً.</p>
        </div>
        """
    elif cpk < 1.33:
        return """
        <div class="engineer-insight" style="background-color: #fcf3cf; border-color: #f1c40f; color: #b7950b;">
            <div class="engineer-title">⚠️ تنبيه هندسي:</div>
            <p><b>العملية مقبولة بحذر (Marginal):</b></p>
            <p>العملية تفي بالمواصفات بالكاد. <b>الإجراء:</b> راقب العملية وحاول تقليل التشتت.</p>
        </div>
        """
    else:
        return """
        <div class="engineer-insight">
            <div class="engineer-title">✅ مصادقة هندسية:</div>
            <p><b>العملية ممتازة (World Class):</b></p>
            <p>العملية مستقرة وتقع في منتصف المواصفات تماماً. استمر على هذا الأداء.</p>
        </div>
        """

def check_data_health(data, col_name):
    """فحص شامل لصحة البيانات"""
    report = []
    # 1. التوزيع الطبيعي
    stat, p_norm = stats.shapiro(data)
    if p_norm > 0.05:
        report.append(f"<div class='success-box'><b>✅ التوزيع طبيعي:</b> البيانات تتبع منحنى الجرس (P={p_norm:.3f}).</div>")
    else:
        report.append(f"<div class='warning-box'><b>⚠️ التوزيع غير طبيعي:</b> البيانات منحرفة (P={p_norm:.3f}).</div>")

    # 2. القيم الشاذة (Outliers)
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    outliers = data[((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR)))]
    
    if len(outliers) == 0:
        report.append("<div class='success-box'><b>✅ لا توجد قيم شاذة:</b> البيانات نظيفة.</div>")
    else:
        report.append(f"<div class='error-box'><b>🚨 تم اكتشاف {len(outliers)} قيم شاذة:</b> قد تشوه النتائج.</div>")
        
    return "".join(report)

def analyze_variance_sources(df, target, factors):
    """تحليل مصادر التشتت ANOVA"""
    try:
        formula = f"{target} ~ " + " + ".join([f"C({f})" for f in factors])
        model = ols(formula, data=df).fit()
        aov_table = sm.stats.anova_lm(model, typ=2)
        total_sum_sq = aov_table['sum_sq'].sum()
        aov_table['Contribution_%'] = (aov_table['sum_sq'] / total_sum_sq) * 100
        res = aov_table[['Contribution_%']].sort_values(by='Contribution_%', ascending=False)
        res.index = [i.replace('C(', '').replace(')', '') for i in res.index]
        return res
    except: return None

# ==========================================
# 🚀 الهيكل الرئيسي (Main Structure)
# ==========================================

# 1. العنوان الرئيسي
st.markdown("""
<div class="hero-header">
    <h1>🚀 المحلل الإحصائي الاستشاري</h1>
    <p>Data Science & Engineering Studio</p>
    <div>
        <span style="background:rgba(255,255,255,0.2); padding:5px 10px; border-radius:15px; font-size:0.8em;">Six Sigma</span>
        <span style="background:rgba(255,255,255,0.2); padding:5px 10px; border-radius:15px; font-size:0.8em;">AutoML</span>
        <span style="background:rgba(255,255,255,0.2); padding:5px 10px; border-radius:15px; font-size:0.8em;">DOE</span>
    </div>
</div>
""", unsafe_allow_html=True)

# 2. القائمة الجانبية
# ==========================================
# 📂 2. القائمة الجانبية (مصدر البيانات)
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2804/2804702.png", width=80)
    st.title("لوحة التحكم")
    st.write("---")
    
    # 1. اختيار طريقة الإدخال
    data_source = st.radio(
        "مصدر البيانات:", 
        ("📂 رفع ملف (Excel/CSV)", "✍️ إدخال يدوي (جدول)", "🎲 بيانات تجريبية")
    )
    
    df = None # تهيئة المتغير
    
    # --- الخيار 1: رفع ملف ---
    if data_source == "📂 رفع ملف (Excel/CSV)":
        uploaded_file = st.file_uploader("اختر الملف:", type=['csv', 'xlsx'])
        if uploaded_file:
            df = load_data(uploaded_file)
            if df is None: st.error("❌ الملف تالف.")
    
    # --- الخيار 2: إدخال يدوي (الميزة الجديدة) ---
    elif data_source == "✍️ إدخال يدوي (جدول)":
        st.info("قم بتعريف الأعمدة أولاً، ثم املأ الجدول.")
        
        # خطوة 1: تعريف الأعمدة
        cols_input = st.text_input("أسماء الأعمدة (افصل بينها بفاصلة):", value="الماكينة, الوزن, الطول")
        columns = [x.strip() for x in cols_input.split(',')]
        
        # خطوة 2: إنشاء جدول فارغ (أو استرجاعه من الذاكرة)
        if 'manual_data' not in st.session_state:
            # ننشئ 5 صفوف فارغة للبداية
            st.session_state.manual_data = pd.DataFrame([[""]*len(columns)]*5, columns=columns)
        
        # تحديث الأعمدة إذا تغيرت
        if list(st.session_state.manual_data.columns) != columns:
             st.session_state.manual_data = pd.DataFrame([[""]*len(columns)]*5, columns=columns)

        # خطوة 3: المحرر التفاعلي
        st.write("▼ املأ البيانات هنا:")
        edited_df = st.data_editor(
            st.session_state.manual_data, 
            num_rows="dynamic", # يسمح بإضافة وحذف الصفوف
            use_container_width=True
        )
        
        # خطوة 4: تنظيف البيانات وتحويل الأرقام
        if not edited_df.empty:
            # محاولة تحويل النصوص إلى أرقام تلقائياً
            for col in edited_df.columns:
                edited_df[col] = pd.to_numeric(edited_df[col], errors='ignore')
            
            # حذف الصفوف الفارغة تماماً
            df = edited_df.dropna(how='all')
            
            # تحديث الذاكرة
            st.session_state.manual_data = edited_df

    # --- الخيار 3: بيانات تجريبية ---
    elif data_source == "🎲 بيانات تجريبية":
        if st.button("توليد بيانات عشوائية"):
            np.random.seed(42)
            data = {
                'الإنتاجية': np.random.normal(100, 15, 100),
                'الرضا': np.random.choice(['عال', 'متوسط', 'منخفض'], 100),
                'الحرارة': np.random.normal(25, 5, 100),
                'الخطأ': np.random.poisson(2, 100)
            }
            data['الإنتاجية'] += np.where(data['الرضا']=='عال', 20, 0)
            df = pd.DataFrame(data)
            st.success("تم توليد البيانات بنجاح!")
        else:
            st.info("اضغط الزر لتوليد البيانات.")

    st.markdown("---")
    st.caption("v4.0 - Engineered for Excellence")

# 3. تحميل البيانات
df = None
if uploaded_file:
    df = load_data(uploaded_file)
    if df is None: st.error("❌ الملف تالف أو غير مدعوم.")
elif use_dummy:
    np.random.seed(42)
    data = {
        'الإنتاجية': np.random.normal(100, 15, 200),
        'الوردية': np.random.choice(['صباحي', 'مسائي'], 200),
        'الماكينة': np.random.choice(['M1', 'M2', 'M3'], 200),
        'درجة_الحرارة': np.random.normal(25, 5, 200),
        'الوزن': np.random.normal(50, 2, 200),
        'العيوب': np.random.poisson(2, 200)
    }
    # إضافة علاقة مصطنعة
    data['الإنتاجية'] = data['الإنتاجية'] + (np.where(data['الماكينة']=='M1', 15, 0))
    df = pd.DataFrame(data)

# ==========================================
# 📱 التطبيق والتبويبات
# ==========================================
if df is not None:
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    tabs = st.tabs([
        "📋 استكشاف (Explore)", 
        "🔎 فحص المصدر والصحة", 
        "⚖️ فروقات (Tests)", 
        "🏭 جودة (Six Sigma)", 
        "🧪 تصميم (DOE)", 
        "🤖 ذكاء (AutoML)", 
        "📏 عينات (Planner)"
    ])

    # --- Tab 1: الاستكشاف ---
    with tabs[0]:
        st.subheader("📊 نظرة عامة")
        c1, c2, c3 = st.columns(3)
        with c1: draw_card("عدد السجلات", df.shape[0], "📂")
        with c2: draw_card("عدد الأعمدة", df.shape[1], "🔢")
        with c3: draw_card("المتغيرات الرقمية", len(num_cols), "#️⃣")
        
        st.markdown("<br>", unsafe_allow_html=True)
        col_main, col_desc = st.columns([2, 1])
        with col_main:
            st.markdown("##### 📈 توزيع البيانات")
            if num_cols:
                target_col = st.selectbox("اختر متغيراً للرسم:", num_cols)
                fig = px.histogram(df, x=target_col, marginal="box", template="plotly_white", color_discrete_sequence=['#2a5298'])
                st.plotly_chart(fig, use_container_width=True)
        with col_desc:
            st.markdown("##### 📝 الإحصاء الوصفي")
            if num_cols:
                st.dataframe(df[target_col].describe(), use_container_width=True)

        st.markdown("---")
        excel_data = convert_df_to_excel(df)
        st.download_button("📥 تحميل النتائج (Excel)", excel_data, "results.xlsx")

    # --- Tab 2: فحص المصدر والصحة ---
    with tabs[1]:
        st.subheader("🔎 فحص صحة البيانات ومصدر التشتت")
        col_health, col_source = st.columns(2)
        
        # 1. فحص الصحة
        with col_health:
            st.markdown("##### 1️⃣ هل البيانات سليمة؟")
            if num_cols:
                check_col = st.selectbox("المتغير للفحص:", num_cols, key="hl_c")
                fig_h = px.histogram(df, x=check_col, marginal="box", template="plotly_white", color_discrete_sequence=['#16a085'])
                st.plotly_chart(fig_h, use_container_width=True)
                st.markdown(check_data_health(df[check_col].dropna(), check_col), unsafe_allow_html=True)
        
        # 2. مصدر التشتت
        with col_source:
            st.markdown("##### 2️⃣ ما هو مصدر الاختلاف؟")
            if len(num_cols)>0 and len(cat_cols)>0:
                t_src = st.selectbox("النتيجة (Y):", num_cols, key="src_y")
                f_src = st.multiselect("العوامل (X):", cat_cols, key="src_x")
                if st.button("تحليل المصدر"):
                    if f_src:
                        res_contrib = analyze_variance_sources(df, t_src, f_src)
                        if res_contrib is not None:
                            fig_pie = px.pie(res_contrib, values='Contribution_%', names=res_contrib.index, title="نسبة المساهمة في التشتت", color_discrete_sequence=px.colors.sequential.Teal)
                            st.plotly_chart(fig_pie, use_container_width=True)
                            
                            top_f = res_contrib.index[0]
                            top_v = res_contrib.iloc[0, 0]
                            st.markdown(f"""
                            <div class="engineer-insight">
                                <div class="engineer-title">💡 كشف المصدر:</div>
                                <p>المسؤول الأكبر عن التغير في <b>{t_src}</b> هو <b>{top_f}</b> بنسبة <b>{top_v:.1f}%</b>.</p>
                            </div>""", unsafe_allow_html=True)
                        else: st.error("بيانات غير كافية.")
                    else: st.warning("اختر عواملاً.")

    # --- Tab 3: الفروقات ---
    with tabs[2]:
        st.subheader("⚖️ اختبار الفروقات (مع العقل الهندسي)")
        if num_cols and cat_cols:
            c1, c2 = st.columns(2)
            y_hyp = c1.selectbox("النتيجة (Y):", num_cols, key='hy_y')
            x_hyp = c2.selectbox("المجموعة (X):", cat_cols, key='hy_x')
            
            if st.button("تحليل الفروقات"):
                groups = df.dropna(subset=[y_hyp, x_hyp])[x_hyp].unique()
                if len(groups) >= 2:
                    g_data = [df[df[x_hyp]==g][y_hyp] for g in groups]
                    if len(groups) == 2:
                        s, p = stats.ttest_ind(g_data[0], g_data[1])
                        t_name = "T-Test"
                    else:
                        s, p = stats.f_oneway(*g_data)
                        t_name = "ANOVA"
                    
                    fig_box = px.box(df, x=x_hyp, y=y_hyp, color=x_hyp, template="plotly_white")
                    st.plotly_chart(fig_box, use_container_width=True)
                    st.markdown(explain_hypothesis(p, t_name), unsafe_allow_html=True)
                    
                    if t_name=="ANOVA" and p<0.05:
                        st.write("نتائج Tukey:")
                        st.text(pairwise_tukeyhsd(df[y_hyp], df[x_hyp]).summary())
                else: st.warning("مجموعتان على الأقل.")

    # --- Tab 4: الجودة ---
    with tabs[3]:
        st.subheader("🏭 ضبط الجودة (Process Capability)")
        if num_cols:
            q_col = st.selectbox("متغير الجودة:", num_cols, key='q_c')
            mean, std = df[q_col].mean(), df[q_col].std()
            c_ctrl, c_cap = st.columns(2)
            
            with c_ctrl:
                st.markdown("##### خريطة التحكم")
                ucl, lcl = mean + 3*std, mean - 3*std
                fig_c = go.Figure()
                fig_c.add_trace(go.Scatter(y=df[q_col], mode='lines+markers', name='Data'))
                fig_c.add_hline(y=ucl, line_color='red', line_dash='dash', annotation_text='UCL')
                fig_c.add_hline(y=lcl, line_color='red', line_dash='dash', annotation_text='LCL')
                fig_c.add_hline(y=mean, line_color='green', annotation_text='Mean')
                fig_c.update_layout(template="plotly_white")
                st.plotly_chart(fig_c, use_container_width=True)
            
            with c_cap:
                st.markdown("##### تحليل القدرة")
                usl = st.number_input("USL:", value=mean + 4*std)
                lsl = st.number_input("LSL:", value=mean - 4*std)
                if usl > lsl:
                    Cpk = min((usl - mean)/(3*std), (mean - lsl)/(3*std))
                    st.metric("Cpk", f"{Cpk:.2f}")
                    st.markdown(explain_capability(Cpk), unsafe_allow_html=True)

    # --- Tab 5: DOE ---
    with tabs[4]:
        st.subheader("🧪 تحليل التفاعل (Interaction Plot)")
        if len(num_cols) >= 1 and (len(cat_cols) + len(num_cols)) >= 2:
            d_y = st.selectbox("النتيجة (Y):", num_cols, key='doe_y')
            remaining = [c for c in df.columns if c!=d_y]
            d_x1 = st.selectbox("العامل 1:", remaining, key='doe_x1')
            d_x2 = st.selectbox("العامل 2:", [c for c in remaining if c!=d_x1], key='doe_x2')
            
            if st.button("رسم التفاعل"):
                try:
                    
                    df_g = df.groupby([d_x1, d_x2])[d_y].mean().reset_index()
                    fig_int = px.line(df_g, x=d_x1, y=d_y, color=d_x2, markers=True, title="Interaction Plot", template="plotly_white")
                    st.plotly_chart(fig_int, use_container_width=True)
                    
                    model = ols(f'{d_y} ~ C({d_x1}) * C({d_x2})', data=df).fit()
                    st.write("**جدول ANOVA للتفاعل:**")
                    st.dataframe(sm.stats.anova_lm(model, typ=2).style.format("{:.4f}"), use_container_width=True)
                except Exception as e: st.error(f"خطأ: {e}")

    # --- Tab 6: AutoML ---
    with tabs[5]:
        st.subheader("🤖 الذكاء الاصطناعي")
        mode = st.radio("اختر:", ["كشف الأهمية (Drivers)", "صائد الشواذ (Anomalies)", "التجميع (Clustering)"], horizontal=True)
        st.markdown("---")
        
        if "الأهمية" in mode:
            t_ml = st.selectbox("الهدف:", num_cols, key='ml_t')
            f_ml = st.multiselect("المؤثرات:", [c for c in num_cols if c!=t_ml], key='ml_f')
            if st.button("تشغيل"):
                if f_ml:
                    rf = RandomForestRegressor(n_estimators=100).fit(df[f_ml].dropna(), df[t_ml].dropna())
                    imp = pd.DataFrame({'Feature': f_ml, 'Importance': rf.feature_importances_}).sort_values('Importance', ascending=False)
                    st.plotly_chart(px.bar(imp, x='Importance', y='Feature', orientation='h', template="plotly_white"), use_container_width=True)
                    st.success(f"أهم عامل هو: {imp.iloc[0,0]}")

        elif "الشواذ" in mode:
            c_iso = st.selectbox("العمود:", num_cols, key='iso')
            if st.button("كشف"):
                iso_d = df[[c_iso]].dropna()
                iso_d['Anomaly'] = IsolationForest(contamination=0.05).fit_predict(iso_d)
                st.plotly_chart(px.scatter(iso_d, y=c_iso, color=iso_d['Anomaly'].astype(str), color_discrete_map={'-1':'red', '1':'blue'}, template="plotly_white"), use_container_width=True)
        
        elif "التجميع" in mode:
            
            c_cl = st.multiselect("الأعمدة:", num_cols, key='cl')
            k = st.slider("K:", 2, 8, 3)
            if st.button("تجميع"):
                if len(c_cl)>=2:
                    X = StandardScaler().fit_transform(df[c_cl].dropna())
                    km = KMeans(n_clusters=k).fit(X)
                    df_c = df[c_cl].dropna()
                    df_c['Cluster'] = km.labels_.astype(str)
                    st.plotly_chart(px.scatter_matrix(df_c, dimensions=c_cl, color='Cluster', template="plotly_white"), use_container_width=True)

    # --- Tab 7: Planner ---
    with tabs[6]:
        st.subheader("📏 مخطط العينات")
        
        cp1, cp2 = st.columns(2)
        conf = cp1.selectbox("الثقة:", [0.90, 0.95, 0.99], index=1)
        err = cp2.number_input("الخطأ (%):", 1.0, 10.0, 5.0) / 100
        z = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}[conf]
        n = (z**2 * 0.5 * 0.5) / (err**2)
        st.markdown(f'<div class="metric-card"><h1>{int(n)+1}</h1><p>حجم العينة المطلوب</p></div>', unsafe_allow_html=True)

else:
    st.info("👋 مرحباً! الرجاء رفع ملف البيانات من القائمة الجانبية.")

