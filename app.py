import streamlit as st
import pandas as pd
import plotly.express as px

# Streamlit Page Config
st.set_page_config(
    page_title="Order Flow Insights - Dynamic Dashboard",
    layout="wide",
    page_icon="📊"
)

st.title("🛒 Order Flow Insights: Universal Ecommerce Analytics Dashboard")

st.markdown("""
Welcome! Upload any **ecommerce orders dataset (CSV)** below to explore insights instantly.  
Your data will be analyzed, cleaned, and visualized right here ⚡
""")

# File uploader
uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])

if uploaded_file:
    # Read the uploaded dataset
    df_raw = pd.read_csv(uploaded_file)
    st.subheader("📄 Raw Data Preview")
    st.dataframe(df_raw.head(10))

    # ===============================
    # 🧹 PREPROCESSING SECTION
    # ===============================
    st.markdown("### 🧹 Data Preprocessing Steps")

    df = df_raw.copy()

    # Handle missing values
    df.fillna({
        "product_category": "Unknown",
        "payment_method": "Unknown",
        "delivery_status": "Pending"
    }, inplace=True)

    # Convert date columns
    date_cols = [col for col in df.columns if "date" in col.lower()]
    if date_cols:
        df[date_cols[0]] = pd.to_datetime(df[date_cols[0]], errors="coerce")
        df.rename(columns={date_cols[0]: "order_date"}, inplace=True)
    else:
        st.error("⚠️ No 'order_date' column found. Please include one in your dataset.")
        st.stop()

    # Create features
    if "quantity" in df.columns and "price" in df.columns:
        df["sales"] = df["quantity"] * df["price"]
    else:
        st.error("⚠️ Dataset must contain 'quantity' and 'price' columns.")
        st.stop()

    df["month"] = df["order_date"].dt.strftime("%b")
    df["year"] = df["order_date"].dt.year

    # Remove price outliers
    df = df[df["price"] < df["price"].quantile(0.99)]

    st.success("✅ Preprocessing Completed Successfully!")

    # Show cleaned data
    st.subheader("🟢 Cleaned Dataset Sample")
    st.dataframe(df.head(10))

    # ===============================
    # 📊 DASHBOARD SECTION
    # ===============================
    st.markdown("---")
    st.header("📊 Interactive Analytics Dashboard")

    # Sidebar filters
    st.sidebar.header("🔍 Filter Options")
    category = st.sidebar.multiselect("Product Category", df['product_category'].unique())
    payment = st.sidebar.multiselect("Payment Method", df['payment_method'].unique())

    df_filtered = df.copy()
    if category:
        df_filtered = df_filtered[df_filtered['product_category'].isin(category)]
    if payment:
        df_filtered = df_filtered[df_filtered['payment_method'].isin(payment)]

    # KPIs
    total_sales = df_filtered['sales'].sum()
    avg_order = df_filtered['sales'].mean()
    orders = df_filtered['order_id'].nunique() if 'order_id' in df_filtered.columns else len(df_filtered)

    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("💰 Total Sales", f"${total_sales:,.2f}")
    kpi2.metric("📦 Average Order Value", f"${avg_order:,.2f}")
    kpi3.metric("🧾 Total Orders", orders)

    st.markdown("---")

    # Charts
    monthly = (
        df_filtered.groupby(df_filtered['order_date'].dt.to_period("M"))
        .sum(numeric_only=True)
        .reset_index()
    )
    monthly['order_date'] = monthly['order_date'].astype(str)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📈 Monthly Sales Trend")
        st.plotly_chart(px.line(monthly, x='order_date', y='sales', markers=True), use_container_width=True)

    with col2:
        st.subheader("🏷️ Top Product Categories")
        cat_sales = df_filtered.groupby('product_category')['sales'].sum().reset_index()
        st.plotly_chart(px.bar(cat_sales, x='product_category', y='sales'), use_container_width=True)

    st.subheader("💳 Payment Method Distribution")
    st.plotly_chart(px.pie(df_filtered, names='payment_method'), use_container_width=True)

    st.markdown("---")
    st.success("🎯 Dashboard Generated Successfully for Uploaded Dataset!")

    # ===============================
    # 🤖 MACHINE LEARNING PREDICTION
    # ===============================
    st.markdown("---")
    st.header("🤖 Predictive Insights ")

    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import accuracy_score, r2_score

    # Use the filtered dataframe from your dashboard
    df_ml = df_filtered.copy()

    # Encode categorical columns
    for col in df_ml.select_dtypes(include='object').columns:
        df_ml[col] = LabelEncoder().fit_transform(df_ml[col].astype(str))

    # Try to predict 'sales' if present
    if 'sales' in df_ml.columns:
        X = df_ml.drop(columns=['sales'])
        y = df_ml['sales']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression()
        # Drop any datetime columns before training
        X_train = X_train.select_dtypes(exclude=['datetime64[ns]'])
        X_test = X_test.select_dtypes(exclude=['datetime64[ns]'])

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        score = r2_score(y_test, preds)

        st.subheader("📊 Regression Model: Predicting Sales")
        st.metric("Model R² Score", f"{score:.3f}")

        if score > 0.8:
            st.success("💡 Strong predictive relationship — your dataset has clear sales-driving factors!")
        elif score > 0.5:
            st.info("📈 Moderate relationship — useful for business trend forecasting.")
        else:
            st.warning("⚠️ Weak relationship — data may need more features or cleaning.")
    else:
        # Classification fallback if 'sales' not found
        target_candidates = [col for col in df_ml.columns if df_ml[col].nunique() <= 10]
        if target_candidates:
            target = target_candidates[-1]
            X = df_ml.drop(columns=[target])
            y = df_ml[target]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = LogisticRegression(max_iter=200)
            # Drop any datetime columns before training
            X_train = X_train.select_dtypes(exclude=['datetime64[ns]'])
            X_test = X_test.select_dtypes(exclude=['datetime64[ns]'])

            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            acc = accuracy_score(y_test, preds)

            st.subheader(f"📊 Classification Model: Predicting '{target}'")
            st.metric("Model Accuracy", f"{acc*100:.2f}%")

            if acc > 0.8:
                st.success("✅ Strong predictive accuracy — excellent pattern recognition in your dataset!")
            elif acc > 0.5:
                st.info("📈 Moderate predictive power — usable for trend classification.")
            else:
                st.warning("⚠️ Weak model — dataset might be too small or unbalanced.")
        else:
            st.error("⚙️ Not enough categorical or numeric data for prediction.")

else:
    st.info("👆 Upload a dataset to begin analyzing.")
