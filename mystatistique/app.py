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
from sklearn.preprocessing import StandardScaler, LabelEncoder
from io import BytesIO

# ==========================================
# ⚙️ إعدادات الصفحة (Page Config)
# ==========================================
st.set_page_config(
    page_title="المحلل الإحصائي الشامل (Pro)", 
    layout="wide", 
    page_icon="📊",
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
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] p {
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
    .hero-header h1 { margin: 0; font-size: 2.5em; font-weight: 800; }
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

    /* 5. صناديق التنبيه */
    .success-box { background-color: #d1e7dd; color: #0f5132; padding: 15px; border-radius: 10px; border-right: 5px solid #198754; margin-bottom: 10px; }
    .warning-box { background-color: #fff3cd; color: #664d03; padding: 15px; border-radius: 10px; border-right: 5px solid #ffc107; margin-bottom: 10px; }
    
    /* 6. تحسين الأزرار */
    .stButton>button {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        color: white; border: none; border-radius: 8px; font-weight: bold;
    }
    .stButton>button:hover { opacity: 0.9; color: white; }

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

def check_normality(data):
    if len(data) < 3: return True
    try:
        stat, p_value = stats.shapiro(data)
        return p_value > 0.05
    except: return True

@st.cache_data
def convert_df_to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
    return output.getvalue()

# ==========================================
# 🚀 الهيكل الرئيسي (Main Structure)
# ==========================================

# 1. العنوان الرئيسي (Hero Section)
st.markdown("""
<div class="hero-header">
    <h1>🚀 المحلل الإحصائي الشامل</h1>
    <p>منصة ذكية للتحليل الإحصائي، ضبط الجودة، وعلوم البيانات</p>
    <div>
        <span style="background:rgba(255,255,255,0.2); padding:5px 10px; border-radius:15px; font-size:0.8em;">Data Science</span>
        <span style="background:rgba(255,255,255,0.2); padding:5px 10px; border-radius:15px; font-size:0.8em;">Six Sigma</span>
        <span style="background:rgba(255,255,255,0.2); padding:5px 10px; border-radius:15px; font-size:0.8em;">AI</span>
    </div>
</div>
""", unsafe_allow_html=True)

# 2. القائمة الجانبية (Sidebar)
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2804/2804702.png", width=80)
    st.title("لوحة التحكم")
    st.write("---")
    uploaded_file = st.file_uploader("📂 رفع ملف البيانات (Excel/CSV)", type=['csv', 'xlsx'])
    
    use_dummy = st.checkbox("استخدام بيانات تجريبية", value=False)
    
    st.markdown("---")
    st.info("💡 **نصيحة:** تأكد أن الصف الأول في الملف يحتوي على أسماء الأعمدة.")

# 3. تحميل البيانات
df = None
if uploaded_file:
    df = load_data(uploaded_file)
    if df is None: st.error("❌ الملف تالف أو غير مدعوم.")
elif use_dummy:
    np.random.seed(42)
    data = {
        'الإنتاجية': np.random.normal(100, 15, 200),
        'الرضا_الوظيفي': np.random.choice(['عال', 'متوسط', 'منخفض'], 200),
        'درجة_الحرارة': np.random.normal(25, 5, 200),
        'الوقت_المستغرق': np.random.normal(40, 10, 200),
        'الخطأ': np.random.poisson(2, 200)
    }
    # إضافة علاقة مصطنعة
    data['الإنتاجية'] = data['الإنتاجية'] + (np.where(data['الرضا_الوظيفي']=='عال', 20, 0))
    df = pd.DataFrame(data)

# ==========================================
# 📱 التطبيق والتبويبات (Tabs)
# ==========================================
if df is not None:
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    # شريط التبويبات
    tabs = st.tabs([
        "📋 استكشاف (Explore)", 
        "⚖️ فروقات (Tests)", 
        "🏭 جودة (Six Sigma)", 
        "🧪 تصميم (DOE)", 
        "🤖 ذكاء (AutoML)", 
        "📏 عينات (Planner)"
    ])

    # --- Tab 1: الاستكشاف ---
    with tabs[0]:
        st.subheader("📊 نظرة عامة")
        
        # البطاقات العلوية
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
                fig = px.histogram(df, x=target_col, marginal="box", 
                                   template="plotly_white", color_discrete_sequence=['#2a5298'])
                st.plotly_chart(fig, use_container_width=True)
        
        with col_desc:
            st.markdown("##### 📝 الإحصاء الوصفي")
            if num_cols:
                desc = df[target_col].describe()
                st.dataframe(desc, use_container_width=True)

        # زر التحميل
        st.markdown("---")
        excel_data = convert_df_to_excel(df)
        st.download_button("📥 تحميل البيانات (Excel)", excel_data, "data_export.xlsx")

    # --- Tab 2: الفروقات ---
    with tabs[1]:
        st.subheader("⚖️ اختبار الفرضيات الآلي")
        if len(num_cols) > 0 and len(cat_cols) > 0:
            c1, c2 = st.columns(2)
            y_var = c1.selectbox("المتغير الرقمي (النتيجة):", num_cols, key='hy_y')
            x_group = c2.selectbox("متغير التجميع (المقارنة):", cat_cols, key='hy_x')
            
            if st.button("🚀 تحليل الفروقات"):
                try:
                    df_sub = df.dropna(subset=[y_var, x_group])
                    groups = df_sub[x_group].unique()
                    group_data = [df_sub[df_sub[x_group] == g][y_var] for g in groups]
                    
                    if len(groups) < 2:
                        st.warning("⚠️ يجب وجود مجموعتين على الأقل للمقارنة.")
                    else:
                        # فحص الافتراضات
                        normality = all([check_normality(g) for g in group_data])
                        s, p_levene = stats.levene(*group_data)
                        homogeneity = p_levene > 0.05
                        
                        # اختيار الاختبار
                        test_name = ""
                        p_val = 1.0
                        
                        if len(groups) == 2:
                            if normality and homogeneity:
                                s, p_val = stats.ttest_ind(group_data[0], group_data[1])
                                test_name = "T-Test (Independent)"
                            elif normality and not homogeneity:
                                s, p_val = stats.ttest_ind(group_data[0], group_data[1], equal_var=False)
                                test_name = "Welch's T-Test"
                            else:
                                s, p_val = stats.mannwhitneyu(group_data[0], group_data[1])
                                test_name = "Mann-Whitney U"
                        else: # > 2 groups
                            if normality and homogeneity:
                                s, p_val = stats.f_oneway(*group_data)
                                test_name = "One-Way ANOVA"
                            else:
                                s, p_val = stats.kruskal(*group_data)
                                test_name = "Kruskal-Wallis"
                        
                        # عرض النتائج
                        st.markdown(f"""
                        <div class="metric-card" style="text-align:right;">
                            <h4>🔍 نتيجة التحليل</h4>
                            <p><b>الاختبار المستخدم:</b> {test_name}</p>
                            <p><b>P-Value:</b> {p_val:.5f}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        if p_val < 0.05:
                            st.markdown('<div class="success-box">✅ <b>يوجد فرق جوهري</b> (ذو دلالة إحصائية) بين المجموعات.</div>', unsafe_allow_html=True)
                            if test_name == "One-Way ANOVA":
                                tukey = pairwise_tukeyhsd(df_sub[y_var], df_sub[x_group])
                                st.write("**المقارنات البعدية (Tukey):**")
                                st.text(tukey.summary())
                        else:
                            st.markdown('<div class="warning-box">✋ <b>لا يوجد فرق جوهري</b> بين المجموعات.</div>', unsafe_allow_html=True)
                        
                        # الرسم
                        fig_box = px.box(df_sub, x=x_group, y=y_var, color=x_group, template="plotly_white")
                        st.plotly_chart(fig_box, use_container_width=True)

                except Exception as e: st.error(f"خطأ: {e}")
        else: st.info("اختر متغيرات رقمية وفئوية.")

    # --- Tab 3: الجودة ---
    with tabs[2]:
        st.subheader("🏭 ضبط الجودة (Six Sigma)")
        if num_cols:
            q_target = st.selectbox("متغير الجودة:", num_cols, key='six_t')
            
            c_ctrl, c_cap = st.columns(2)
            
            # Control Chart
            with c_ctrl:
                st.markdown("##### 1️⃣ خريطة التحكم (I-MR)")
                data_q = df[q_target]
                mean_q, std_q = data_q.mean(), data_q.std()
                ucl, lcl = mean_q + 3*std_q, mean_q - 3*std_q
                
                fig_c = go.Figure()
                fig_c.add_trace(go.Scatter(y=data_q, mode='lines+markers', name='Data', line=dict(color='#2a5298')))
                fig_c.add_hline(y=ucl, line_dash='dash', line_color='red', annotation_text='UCL')
                fig_c.add_hline(y=lcl, line_dash='dash', line_color='red', annotation_text='LCL')
                fig_c.add_hline(y=mean_q, line_color='green', annotation_text='Mean')
                fig_c.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(fig_c, use_container_width=True)
                
            # Capability
            with c_cap:
                st.markdown("##### 2️⃣ قدرة العملية (Capability)")
                usl = st.number_input("USL (الحد الأعلى):", value=mean_q + 4*std_q)
                lsl = st.number_input("LSL (الحد الأدنى):", value=mean_q - 4*std_q)
                
                if usl > lsl:
                    Cp = (usl - lsl) / (6 * std_q)
                    Cpk = min((usl - mean_q)/(3*std_q), (mean_q - lsl)/(3*std_q))
                    
                    cc1, cc2 = st.columns(2)
                    with cc1: draw_card("Cp", f"{Cp:.2f}", "📐")
                    with cc2: draw_card("Cpk", f"{Cpk:.2f}", "🎯")
                    
                    if Cpk < 1.33: st.warning("⚠️ العملية تحتاج تحسين")
                    else: st.success("✅ العملية ممتازة")

    # --- Tab 4: DOE ---
    with tabs[3]:
        st.subheader("🧪 تحليل التفاعل (Interaction)")
        if len(num_cols) >= 1 and (len(num_cols) + len(cat_cols)) >= 3:
            d_y = st.selectbox("النتيجة (Y):", num_cols, key='doe_y')
            d_x1 = st.selectbox("العامل 1:", [c for c in df.columns if c!=d_y], key='doe_x1')
            d_x2 = st.selectbox("العامل 2:", [c for c in df.columns if c!=d_y and c!=d_x1], key='doe_x2')
            
            if st.button("تحليل التفاعل"):
                try:
                    df_doe = df.groupby([d_x1, d_x2])[d_y].mean().reset_index()
                    fig_int = px.line(df_doe, x=d_x1, y=d_y, color=d_x2, markers=True, 
                                      title="Interaction Plot", template="plotly_white")
                    st.plotly_chart(fig_int, use_container_width=True)
                    
                    # ANOVA Model
                    model = ols(f'{d_y} ~ C({d_x1}) * C({d_x2})', data=df).fit()
                    aov_table = sm.stats.anova_lm(model, typ=2)
                    st.write("**جدول ANOVA للتفاعل:**")
                    st.dataframe(aov_table.style.format("{:.4f}"), use_container_width=True)
                    
                except Exception as e: st.error(f"حدث خطأ: {e}")

    # --- Tab 5: AutoML ---
    with tabs[4]:
        st.subheader("🤖 الذكاء الاصطناعي")
        
        mode = st.radio("اختر الوظيفة:", ["🌳 كشف المؤثرات (Feature Importance)", "🔍 صائد الشواذ (Anomaly Detection)", "🧩 التجميع (Clustering)"], horizontal=True)
        st.markdown("---")
        
        if "المؤثرات" in mode:
            t_ml = st.selectbox("الهدف (Y):", num_cols, key='rf_y')
            f_ml = st.multiselect("المؤثرات (X):", [c for c in num_cols if c!=t_ml], key='rf_x')
            if st.button("تشغيل Random Forest"):
                if f_ml:
                    with st.spinner("جاري التحليل..."):
                        df_m = df[f_ml + [t_ml]].dropna()
                        rf = RandomForestRegressor(n_estimators=100)
                        rf.fit(df_m[f_ml], df_m[t_ml])
                        res = pd.DataFrame({'Feature': f_ml, 'Importance': rf.feature_importances_}).sort_values('Importance', ascending=False)
                        fig_imp = px.bar(res, x='Importance', y='Feature', orientation='h', template="plotly_white", color='Importance')
                        st.plotly_chart(fig_imp, use_container_width=True)
                else: st.warning("اختر مؤثرات.")

        elif "الشواذ" in mode:
            col_iso = st.selectbox("العمود للفحص:", num_cols, key='iso_c')
            contam = st.slider("نسبة الشك:", 0.01, 0.2, 0.05)
            if st.button("كشف"):
                iso_data = df[[col_iso]].dropna()
                clf = IsolationForest(contamination=contam).fit(iso_data)
                iso_data['Status'] = clf.predict(iso_data)
                iso_data['Status'] = iso_data['Status'].map({1: 'Normal', -1: 'Anomaly'})
                
                fig_iso = px.scatter(iso_data, y=col_iso, color='Status', 
                                     color_discrete_map={'Normal':'#2a5298', 'Anomaly':'#dc3545'}, template="plotly_white")
                st.plotly_chart(fig_iso, use_container_width=True)
                
        elif "التجميع" in mode:
            c_clust = st.multiselect("اختر أعمدة للتجميع:", num_cols, key='km_c')
            k = st.slider("عدد المجموعات (K):", 2, 8, 3)
            if st.button("تجميع"):
                if len(c_clust) >= 2:
                    X = df[c_clust].dropna()
                    X_sc = StandardScaler().fit_transform(X)
                    km = KMeans(n_clusters=k).fit(X_sc)
                    X['Cluster'] = km.labels_.astype(str)
                    fig_k = px.scatter_matrix(X, dimensions=c_clust, color='Cluster', template="plotly_white")
                    st.plotly_chart(fig_k, use_container_width=True)

    # --- Tab 6: Planner ---
    with tabs[5]:
        st.subheader("📏 مخطط حجم العينة")
        cp1, cp2 = st.columns(2)
        conf_lvl = cp1.selectbox("مستوى الثقة:", [0.90, 0.95, 0.99], index=1)
        marg_err = cp2.number_input("هامش الخطأ (%):", 1.0, 10.0, 5.0) / 100
        
        z_score = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}[conf_lvl]
        n_size = (z_score**2 * 0.5 * 0.5) / (marg_err**2)
        
        st.markdown(f"""
        <div style="background:#e3f2fd; padding:30px; border-radius:15px; text-align:center; border:2px solid #2a5298;">
            <h2 style="color:#1e3c72; margin:0;">حجم العينة المطلوب</h2>
            <h1 style="font-size:4em; color:#2a5298; margin:10px 0;">{int(n_size)+1}</h1>
            <p>لتحقيق ثقة {conf_lvl*100:.0f}%</p>
        </div>
        """, unsafe_allow_html=True)

else:
    st.info("👋 مرحباً! ابدأ برفع ملف البيانات من القائمة الجانبية.")
