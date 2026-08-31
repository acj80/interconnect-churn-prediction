import pandas as pd
import requests
import streamlit as st


API_BASE_URL = "http://127.0.0.1:8000"


# ==================================================
# Configuration
# ==================================================

st.set_page_config(
    page_title="Interconnect Churn Dashboard",
    page_icon="📊",
    layout="wide",
)


# ==================================================
# API utilities
# ==================================================

def get_api_status() -> bool:
    try:
        response = requests.get(
            f"{API_BASE_URL}/health",
            timeout=5,
        )

        return (
            response.status_code == 200
            and response.json().get("model_loaded") is True
        )

    except requests.exceptions.RequestException:
        return False


def request_prediction(payload: dict) -> dict | None:
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json=payload,
            timeout=10,
        )

        if response.status_code == 200:
            return response.json()

        st.error(
            f"API returned status code {response.status_code}."
        )

        st.json(response.json())

    except requests.exceptions.RequestException as error:
        st.error(
            "An error occurred while contacting "
            "the prediction API."
        )

        st.exception(error)

    return None

# ==================================================
# Feature engineering utilities
# ==================================================


def build_derived_features(
    internet_service: str,
    online_security: str,
    online_backup: str,
    device_protection: str,
    tech_support: str,
    streaming_tv: str,
    streaming_movies: str,
    has_phone: int,
    payment_method: str,
) -> dict:
    has_internet = int(
        internet_service != "No internet service"
    )

    internet_add_on_count = sum(
        value == "Yes"
        for value in [
            online_security,
            online_backup,
            device_protection,
            tech_support,
        ]
    )

    streaming_count = sum(
        value == "Yes"
        for value in [
            streaming_tv,
            streaming_movies,
        ]
    )

    has_tech_protection = int(
        tech_support == "Yes"
        or device_protection == "Yes"
    )

    automatic_payment = int(
        payment_method
        in [
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ]
    )

    return {
        "HasInternet": has_internet,
        "HasPhone": has_phone,
        "InternetAddOnCount": internet_add_on_count,
        "StreamingCount": streaming_count,
        "HasTechProtection": has_tech_protection,
        "AutomaticPayment": automatic_payment,
    }

# ==================================================
# Sidebar
# ==================================================


def render_sidebar(api_available: bool) -> None:

    with st.sidebar:
        st.title("Interconnect")

        st.markdown(
            "Customer Churn Prediction"
        )

        st.divider()

        st.markdown("### Application")

        st.write(
            "Machine Learning dashboard for churn risk "
            "estimation and model monitoring."
        )

        st.divider()

        st.markdown("### Model")

        st.write(
            "CatBoostClassifier + One-Hot Encoding"
        )

        st.write(
            "Official threshold: 0.5"
        )

        st.divider()

        st.markdown("### Service Status")

        if api_available:
            st.success("API online · Model loaded")

        else:
            st.error("API offline")


# ==================================================
# Header
# ==================================================

def render_header(api_available: bool) -> None:

    st.title(
        "Interconnect — Customer Churn Intelligence"
    )

    st.markdown(
        """
        Predictive analytics dashboard for identifying customers
        with elevated churn risk and supporting retention prioritization.
        """
    )

    st.caption(
        "Customer churn risk estimation using the official "
        "CatBoost production pipeline."
    )

    if api_available:
        st.success(
            "Prediction service available"
        )

    else:
        st.error(
            "FastAPI service is not available. "
            "Start the API before making predictions."
        )


# ==================================================
# Executive overview
# ==================================================

def render_executive_overview() -> None:

    st.divider()
    st.header("1. Executive Overview")

    st.markdown(
        """
        The model estimates each customer's probability of churn.
        The resulting score can support retention teams by helping
        prioritize customers for further review or intervention.
        """
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "CV AUC-ROC",
            "0.8506",
            help=(
                "Average ROC-AUC obtained during cross-validation "
                "on the training set."
            ),
        )

    with col2:
        st.metric(
            "Test AUC-ROC",
            "0.8440",
            help=(
                "ROC-AUC evaluated on the held-out test set."
            ),
        )

    with col3:
        st.metric(
            "Test Accuracy",
            "0.8077",
            help=(
                "Proportion of correct predictions "
                "on the test set."
            ),
        )

    with col4:
        st.metric(
            "Recall — Churn",
            "0.53",
            help=(
                "Proportion of actual churn customers "
                "correctly identified by the model."
            ),
        )

    st.info(
        "The model is intended to support customer retention "
        "prioritization using churn probability as a risk score. "
        "Predictions should support, not replace, business judgment."
    )


# ==================================================
# Model performance
# ==================================================

def render_model_performance() -> None:

    st.divider()
    st.header("2. Model Performance")

    st.markdown(
        "### Test Set Confusion Matrix"
    )

    confusion_df = pd.DataFrame(
        {
            "Predicted No Churn": [938, 174],
            "Predicted Churn": [97, 200],
        },
        index=[
            "Actual No Churn",
            "Actual Churn",
        ],
    )

    st.dataframe(
        confusion_df,
        use_container_width=True,
    )

    st.caption(
        "Of 374 actual churn customers in the test set, "
        "the model correctly identified 200 and missed 174 "
        "using the official 0.5 threshold."
    )

    st.markdown(
        "### Main Predictive Drivers"
    )

    feature_importance = pd.DataFrame(
        {
            "Feature": [
                "Type",
                "TotalCharges",
                "InternetService",
                "MonthlyCharges",
                "OnlineSecurity",
                "TechSupport",
                "PaymentMethod",
            ],
            "Importance": [
                28.12,
                21.52,
                14.05,
                6.86,
                4.49,
                4.28,
                4.24,
            ],
        }
    )

    st.bar_chart(
        feature_importance.set_index("Feature")
    )

    st.caption(
        "Feature importance reflects predictive relevance, "
        "not causal impact."
    )

    st.markdown(
        "### Decision Threshold"
    )

    st.write(
        "The official evaluation threshold is **0.5**. "
        "This threshold is used for the binary prediction "
        "shown by the application."
    )

    st.caption(
        "A production retention strategy may use a different "
        "threshold depending on campaign cost, customer value, "
        "budget and operational capacity."
    )


# ==================================================
# Prediction form
# ==================================================

def render_prediction_form() -> tuple[bool, dict]:

    st.divider()
    st.header("3. Customer Risk Prediction")

    st.write(
        "Enter the customer's current profile to estimate "
        "the probability of churn."
    )

    with st.form("customer_prediction_form"):

        st.markdown(
            "### Contract and Billing"
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            contract_type = st.selectbox(
                "Contract type",
                [
                    "Month-to-month",
                    "One year",
                    "Two year",
                ],
            )

        with col2:
            paperless_billing = st.selectbox(
                "Paperless billing",
                ["Yes", "No"],
            )

        with col3:
            payment_method = st.selectbox(
                "Payment method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
            )

        col1, col2 = st.columns(2)

        with col1:
            monthly_charges = st.number_input(
                "Monthly charges",
                min_value=0.0,
                value=50.0,
                step=1.0,
            )

        with col2:
            total_charges = st.number_input(
                "Total charges",
                min_value=0.0,
                value=500.0,
                step=10.0,
            )

        st.markdown(
            "### Customer Profile"
        )

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            gender = st.selectbox(
                "Gender",
                ["Male", "Female"],
            )

        with col2:
            senior_citizen = st.selectbox(
                "Senior citizen",
                [0, 1],
                format_func=lambda x: (
                    "Yes" if x == 1 else "No"
                ),
            )

        with col3:
            partner = st.selectbox(
                "Partner",
                ["Yes", "No"],
            )

        with col4:
            dependents = st.selectbox(
                "Dependents",
                ["Yes", "No"],
            )

        st.markdown(
            "### Internet Service"
        )

        internet_service = st.selectbox(
            "Internet service",
            [
                "DSL",
                "Fiber optic",
                "No internet service",
            ],
        )

        has_internet = int(
            internet_service != "No internet service"
        )

        if has_internet:

            internet_options = ["Yes", "No"]

            col1, col2, col3 = st.columns(3)

            with col1:
                online_security = st.selectbox(
                    "Online security",
                    internet_options,
                )

                online_backup = st.selectbox(
                    "Online backup",
                    internet_options,
                )

            with col2:
                device_protection = st.selectbox(
                    "Device protection",
                    internet_options,
                )

                tech_support = st.selectbox(
                    "Tech support",
                    internet_options,
                )

            with col3:
                streaming_tv = st.selectbox(
                    "Streaming TV",
                    internet_options,
                )

                streaming_movies = st.selectbox(
                    "Streaming movies",
                    internet_options,
                )

        else:
            online_security = "No internet service"
            online_backup = "No internet service"
            device_protection = "No internet service"
            tech_support = "No internet service"
            streaming_tv = "No internet service"
            streaming_movies = "No internet service"

            st.info(
                "Internet add-on fields are automatically "
                "set to 'No internet service'."
            )

        st.markdown(
            "### Phone Service"
        )

        has_phone = st.selectbox(
            "Phone service",
            [1, 0],
            format_func=lambda x: (
                "Yes" if x == 1 else "No"
            ),
        )

        if has_phone:

            multiple_lines = st.selectbox(
                "Multiple lines",
                ["Yes", "No"],
            )

        else:
            multiple_lines = "No phone service"

            st.info(
                "MultipleLines is automatically set to "
                "'No phone service'."
            )

        derived_features = build_derived_features(
            internet_service=internet_service,
            online_security=online_security,
            online_backup=online_backup,
            device_protection=device_protection,
            tech_support=tech_support,
            streaming_tv=streaming_tv,
            streaming_movies=streaming_movies,
            has_phone=has_phone,
            payment_method=payment_method,
        )

        submitted = st.form_submit_button(
            "Predict churn risk",
            type="primary",
            use_container_width=True,
        )

    payload = {
        "Type": contract_type,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "MultipleLines": multiple_lines,
        "HasInternet": derived_features["HasInternet"],
        "HasPhone": derived_features["HasPhone"],
        "InternetAddOnCount": derived_features["InternetAddOnCount"],
        "StreamingCount": derived_features["StreamingCount"],
        "HasTechProtection": derived_features["HasTechProtection"],
        "AutomaticPayment": derived_features["AutomaticPayment"],
    }

    return submitted, payload


# ==================================================
# Prediction result
# ==================================================

def render_prediction_result(
    result: dict,
    payload: dict,
) -> None:

    probability = result[
        "churn_probability"
    ]

    prediction = result[
        "prediction"
    ]

    risk_level = result[
        "risk_level"
    ]

    st.divider()
    st.subheader("Prediction Result")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Churn probability",
            f"{probability:.1%}",
        )

    with col2:
        st.metric(
            "Predicted class",
            (
                "Churn"
                if prediction == 1
                else "No churn"
            ),
        )

    with col3:
        st.metric(
            "Risk level",
            risk_level.upper(),
        )

    st.progress(
        min(
            max(probability, 0.0),
            1.0,
        )
    )

    st.markdown(
        "### Risk Interpretation"
    )

    if risk_level == "high":

        st.warning(
            "High churn risk. Consider prioritizing "
            "this customer for retention review."
        )

    elif risk_level == "medium":

        st.info(
            "Moderate churn risk. Additional customer "
            "context may help determine whether "
            "intervention is appropriate."
        )

    else:

        st.success(
            "Low predicted churn risk under "
            "the current model."
        )

    with st.expander(
        "View model input"
    ):
        st.json(payload)


# ==================================================
# Footer
# ==================================================

def render_footer() -> None:

    st.divider()

    st.caption(
        "Model outputs represent predictive associations, "
        "not causal effects. Retention actions should be "
        "validated through controlled experiments."
    )


# ==================================================
# Main application
# ==================================================

def main() -> None:

    api_available = get_api_status()

    render_sidebar(
        api_available
    )

    render_header(
        api_available
    )

    render_executive_overview()

    render_model_performance()

    submitted, payload = (
        render_prediction_form()
    )

    if submitted:

        if not api_available:

            st.error(
                "Prediction unavailable because "
                "the API is offline."
            )

        else:

            result = request_prediction(
                payload
            )

            if result is not None:

                render_prediction_result(
                    result,
                    payload,
                )

    render_footer()


if __name__ == "__main__":
    main()
