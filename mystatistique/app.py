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
# ⚙️ إعدادات الصفحة والتصميم
# ==========================================
st.set_page_config(page_title="المحلل الإحصائي الشامل (Pro)", layout="wide", page_icon="📊")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;700&display=swap');
    html, body, [class*="css"] { font-family: 'Cairo', sans-serif; direction: rtl; text-align: right; }
    .header-box { background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); padding: 20px; border-radius: 10px; text-align: center; color: white; margin-bottom: 20px;}
    .success-box { background-color: #d4edda; color: #155724; padding: 10px; border-radius: 5px; border-right: 5px solid #28a745; margin-bottom: 10px; }
    .warning-box { background-color: #fff3cd; color: #856404; padding: 10px; border-radius: 5px; border-right: 5px solid #ffc107; margin-bottom: 10px; }
    .error-box { background-color: #f8d7da; color: #721c24; padding: 10px; border-radius: 5px; border-right: 5px solid #dc3545; margin-bottom: 10px; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 🛠️ دوال المساعدة (Optimized & Cached)
# ==========================================

@st.cache_data(ttl=3600) # يحفظ البيانات في الذاكرة لمدة ساعة لتسريع الأداء
def load_data(file):
    try:
        if file.name.endswith('.csv'):
            return pd.read_csv(file)
        else:
            return pd.read_excel(file, engine='openpyxl')
    except Exception as e:
        return None

@st.cache_data
def convert_df_to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
    return output.getvalue()

def check_normality(data):
    if len(data) < 3: return True # تجاوز للعينات الصغيرة جداً
    stat, p_value = stats.shapiro(data)
    return p_value > 0.05

# ==========================================
# 🚀 الواجهة الرئيسية
# ==========================================
st.markdown('<div class="header-box"><h1>📊 المحلل الإحصائي الشامل (AI & Six Sigma)</h1><p>بديل Minitab: إحصاء، جودة، وتعلّم آلة في مكان واحد</p></div>', unsafe_allow_html=True)

# --- القائمة الجانبية ---
with st.sidebar:
    st.header("📂 البيانات")
    uploaded_file = st.file_uploader("ارفع ملف Excel/CSV", type=['csv', 'xlsx'])
    
    st.markdown("---")
    st.info("💡 نصيحة: تأكد أن الصف الأول يحتوي على أسماء الأعمدة.")

# --- تحميل البيانات ---
df = None
if uploaded_file:
    df = load_data(uploaded_file)
    if df is None:
        st.error("❌ حدث خطأ أثناء قراءة الملف. تأكد أنه سليم.")
else:
    # بيانات تجريبية (Fallback)
    if st.checkbox("تجربة ببيانات وهمية؟"):
        np.random.seed(42)
        data = {
            'الإنتاجية': np.random.normal(100, 10, 150),
            'درجة_الحرارة': np.random.choice(['عالي', 'منخفض'], 150),
            'الضغط': np.random.choice(['عالي', 'منخفض'], 150),
            'الوقت': np.random.normal(50, 5, 150),
            'العيوب': np.random.poisson(2, 150)
        }
        data['الإنتاجية'] = data['الإنتاجية'] + (np.where(data['درجة_الحرارة']=='عالي', 10, 0))
        df = pd.DataFrame(data)

# ==========================================
# 📱 التطبيق (Tabs)
# ==========================================
if df is not None:
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    tabs = st.tabs([
        "📋 استكشاف (Explore)", 
        "⚖️ فروقات (Tests)", 
        "🏭 جودة (Six Sigma)", 
        "🧪 تصميم (DOE)", 
        "🤖 ذكاء (AutoML)", 
        "📏 عينات (Planner)"
    ])

    # -------------------------------------------------------------------------
    # Tab 1: الاستكشاف (Descriptive)
    # -------------------------------------------------------------------------
    with tabs[0]:
        st.subheader("نظرة عامة على البيانات")
        c1, c2 = st.columns([3, 1])
        c1.dataframe(df.head(), use_container_width=True)
        c2.metric("عدد السجلات", df.shape[0])
        c2.metric("عدد الأعمدة", df.shape[1])
        
        st.markdown("---")
        if num_cols:
            col_target = st.selectbox("اختر عموداً لرسم توزيعه:", num_cols)
            c_chart, c_desc = st.columns([2, 1])
            with c_chart:
                fig = px.histogram(df, x=col_target, marginal="box", title=f"توزيع {col_target}", color_discrete_sequence=['#2a5298'])
                st.plotly_chart(fig, use_container_width=True)
            with c_desc:
                desc = df[col_target].describe()
                st.dataframe(desc, use_container_width=True)

    # -------------------------------------------------------------------------
    # Tab 2: الفروقات (Hypothesis Tests)
    # -------------------------------------------------------------------------
    with tabs[1]:
        st.subheader("تحليل الفروقات الآلي")
        if len(num_cols) > 0 and len(cat_cols) > 0:
            y_var = st.selectbox("المتغير الرقمي (Y):", num_cols, key='t2_y')
            x_group = st.selectbox("المجموعة (X):", cat_cols, key='t2_x')
            
            if st.button("🚀 تشغيل الاختبار"):
                try:
                    df_sub = df.dropna(subset=[y_var, x_group])
                    groups = df_sub[x_group].unique()
                    group_data = [df_sub[df_sub[x_group] == g][y_var] for g in groups]
                    
                    if len(groups) < 2:
                        st.warning("تحتاج لمجموعتين على الأقل.")
                    else:
                        # فحص الافتراضات
                        is_normal = all([check_normality(g) for g in group_data])
                        stat_var, p_var = stats.levene(*group_data)
                        is_homo = p_var > 0.05
                        
                        st.write(f"**فحص الافتراضات:** {'✅ طبيعي' if is_normal else '⚠️ غير طبيعي'} | {'✅ تباين متجانس' if is_homo else '⚠️ تباين غير متجانس'}")
                        
                        # اختيار الاختبار
                        p_final = 0
                        test_name = ""
                        
                        if len(groups) == 2:
                            if is_normal and is_homo:
                                stat, p_final = stats.ttest_ind(group_data[0], group_data[1])
                                test_name = "T-Test"
                            elif is_normal and not is_homo:
                                stat, p_final = stats.ttest_ind(group_data[0], group_data[1], equal_var=False)
                                test_name = "Welch T-Test"
                            else:
                                stat, p_final = stats.mannwhitneyu(group_data[0], group_data[1])
                                test_name = "Mann-Whitney U"
                        else: # > 2 groups
                            if is_normal and is_homo:
                                stat, p_final = stats.f_oneway(*group_data)
                                test_name = "ANOVA"
                            else:
                                stat, p_final = stats.kruskal(*group_data)
                                test_name = "Kruskal-Wallis"
                        
                        st.info(f"الاختبار المختار: **{test_name}**")
                        
                        if p_final < 0.05:
                            st.markdown(f'<div class="success-box">✅ توجد فروقات جوهرية (P-value = {p_final:.4f})</div>', unsafe_allow_html=True)
                            if len(groups) > 2 and test_name == "ANOVA":
                                tukey = pairwise_tukeyhsd(endog=df_sub[y_var], groups=df_sub[x_group], alpha=0.05)
                                st.text("نتائج المقارنات البعدية (Tukey):")
                                st.text(tukey.summary())
                        else:
                            st.markdown(f'<div class="warning-box">✋ لا توجد فروقات جوهرية (P-value = {p_final:.4f})</div>', unsafe_allow_html=True)
                            
                        fig_box = px.box(df_sub, x=x_group, y=y_var, color=x_group)
                        st.plotly_chart(fig_box, use_container_width=True)
                except Exception as e:
                    st.error(f"خطأ: {e}")

    # -------------------------------------------------------------------------
    # Tab 3: الجودة (Six Sigma)
    # -------------------------------------------------------------------------
    with tabs[2]:
        st.subheader("ضبط الجودة (Control Charts & Cpk)")
        if num_cols:
            q_col = st.selectbox("متغير الجودة:", num_cols, key="q_c")
            c1, c2 = st.columns(2)
            
            # Control Chart
            with c1:
                data_q = df[q_col]
                mean_q, std_q = data_q.mean(), data_q.std()
                ucl, lcl = mean_q + 3*std_q, mean_q - 3*std_q
                
                fig_c = go.Figure()
                fig_c.add_trace(go.Scatter(y=data_q, mode='lines+markers', name='Data'))
                fig_c.add_hline(y=ucl, line_color='red', line_dash='dash', annotation_text='UCL')
                fig_c.add_hline(y=lcl, line_color='red', line_dash='dash', annotation_text='LCL')
                fig_c.add_hline(y=mean_q, line_color='green', annotation_text='Mean')
                fig_c.update_layout(title="Control Chart (I-MR)")
                st.plotly_chart(fig_c, use_container_width=True)
            
            # Capability
            with c2:
                usl = st.number_input("USL (الحد الأعلى):", value=mean_q + 4*std_q)
                lsl = st.number_input("LSL (الحد الأدنى):", value=mean_q - 4*std_q)
                if usl > lsl:
                    Cp = (usl - lsl) / (6 * std_q)
                    Cpk = min((usl - mean_q)/(3*std_q), (mean_q - lsl)/(3*std_q))
                    st.metric("Cpk Value", f"{Cpk:.2f}")
                    if Cpk < 1.33: st.error("العملية غير قادرة (Low Capability)")
                    else: st.success("العملية ممتازة (High Capability)")

    # -------------------------------------------------------------------------
    # Tab 4: تصميم التجارب (DOE)
    # -------------------------------------------------------------------------
    with tabs[3]:
        st.subheader("تحليل التفاعل (Interaction Plot)")
        if len(num_cols) > 0 and (len(cat_cols) + len(num_cols)) >= 2:
            doe_y = st.selectbox("النتيجة (Y):", num_cols, key='doe_y')
            doe_x1 = st.selectbox("العامل 1:", [c for c in df.columns if c!=doe_y], key='doe_x1')
            doe_x2 = st.selectbox("العامل 2:", [c for c in df.columns if c!=doe_y and c!=doe_x1], key='doe_x2')
            
            if st.button("تحليل التفاعل"):
                try:
                    df_doe = df.groupby([doe_x1, doe_x2])[doe_y].mean().reset_index()
                    fig_int = px.line(df_doe, x=doe_x1, y=doe_y, color=doe_x2, markers=True, title=f"Interaction: {doe_x1} * {doe_x2}")
                    st.plotly_chart(fig_int, use_container_width=True)
                    
                    # ANOVA Model
                    model = ols(f'{doe_y} ~ C({doe_x1}) * C({doe_x2})', data=df).fit()
                    anova_t = sm.stats.anova_lm(model, typ=2)
                    st.write("**جدول ANOVA:**")
                    st.dataframe(anova_t.style.format("{:.4f}"), use_container_width=True)
                except Exception as e:
                    st.error(f"تأكد من اختيار عوامل مناسبة. الخطأ: {e}")

    # -------------------------------------------------------------------------
    # Tab 5: الذكاء الاصطناعي (AutoML)
    # -------------------------------------------------------------------------
    with tabs[4]:
        st.subheader("الذكاء الاصطناعي (Machine Learning)")
        ml_type = st.radio("اختر نوع التحليل:", ["كشف الأهمية (Driver Analysis)", "صائد الشواذ (Anomalies)", "التجميع (Clustering)"], horizontal=True)
        
        if ml_type == "كشف الأهمية (Driver Analysis)":
            target_ml = st.selectbox("الهدف (Target):", num_cols, key='ml_t')
            feats_ml = st.multiselect("المؤثرات (Features):", [c for c in num_cols if c!=target_ml], key='ml_f')
            
            if st.button("تشغيل Random Forest"):
                if feats_ml:
                    with st.spinner("جاري تدريب النموذج..."):
                        df_ml = df[feats_ml + [target_ml]].dropna()
                        rf = RandomForestRegressor(n_estimators=100)
                        rf.fit(df_ml[feats_ml], df_ml[target_ml])
                        imp = pd.DataFrame({'Feature': feats_ml, 'Importance': rf.feature_importances_}).sort_values('Importance', ascending=False)
                        fig_imp = px.bar(imp, x='Importance', y='Feature', orientation='h', title="أهمية العوامل")
                        st.plotly_chart(fig_imp, use_container_width=True)
                else: st.warning("اختر مؤثرات.")

        elif ml_type == "صائد الشواذ (Anomalies)":
            col_iso = st.selectbox("العمود للفحص:", num_cols, key='iso_c')
            if st.button("كشف الشواذ"):
                data_iso = df[[col_iso]].dropna()
                iso = IsolationForest(contamination=0.05).fit(data_iso)
                data_iso['Anomaly'] = iso.predict(data_iso)
                fig_iso = px.scatter(data_iso, y=col_iso, color=data_iso['Anomaly'].astype(str), color_discrete_map={'-1':'red', '1':'blue'})
                st.plotly_chart(fig_iso, use_container_width=True)
                st.write(f"عدد الشواذ: {len(data_iso[data_iso['Anomaly']==-1])}")
        
        elif ml_type == "التجميع (Clustering)":
            clust_cols = st.multiselect("اختر أعمدة للتجميع:", num_cols, key='cl_c')
            k = st.slider("عدد المجموعات:", 2, 8, 3)
            if st.button("تجميع"):
                if len(clust_cols) >= 2:
                    X = df[clust_cols].dropna()
                    X_scaled = StandardScaler().fit_transform(X)
                    kmeans = KMeans(n_clusters=k).fit(X_scaled)
                    X['Cluster'] = kmeans.labels_.astype(str)
                    fig_clust = px.scatter_matrix(X, dimensions=clust_cols, color='Cluster')
                    st.plotly_chart(fig_clust, use_container_width=True)
                else: st.warning("اختر عمودين على الأقل.")

    # -------------------------------------------------------------------------
    # Tab 6: حاسبة العينة (Planning)
    # -------------------------------------------------------------------------
    with tabs[5]:
        st.subheader("تخطيط حجم العينة")
        c1, c2 = st.columns(2)
        conf = c1.selectbox("مستوى الثقة:", [0.90, 0.95, 0.99], index=1)
        err = c2.number_input("هامش الخطأ (%):", 1.0, 10.0, 5.0) / 100
        z = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}[conf]
        n = (z**2 * 0.5 * 0.5) / (err**2)
        st.markdown(f'<div class="success-box" style="text-align:center"><h1>{int(n)+1}</h1><p>حجم العينة المطلوب</p></div>', unsafe_allow_html=True)

else:
    st.info("👈 يرجى رفع ملف البيانات من القائمة الجانبية للبدء.")