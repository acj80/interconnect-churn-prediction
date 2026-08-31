from pydantic import BaseModel, Field, model_validator


class CustomerFeatures(BaseModel):
    Type: str
    PaperlessBilling: str
    PaymentMethod: str

    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)

    gender: str
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: str
    Dependents: str

    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    MultipleLines: str

    HasInternet: int = Field(ge=0, le=1)
    HasPhone: int = Field(ge=0, le=1)

    InternetAddOnCount: int = Field(ge=0)
    StreamingCount: int = Field(ge=0)

    HasTechProtection: int = Field(ge=0, le=1)
    AutomaticPayment: int = Field(ge=0, le=1)

    @model_validator(mode="after")
    def validate_business_rules(self):
        # Phone consistency
        if self.HasPhone == 0 and self.MultipleLines != "No phone service":
            raise ValueError(
                "If HasPhone is 0, MultipleLines must be 'No phone service'."
            )

        if self.HasPhone == 1 and self.MultipleLines == "No phone service":
            raise ValueError(
                "If HasPhone is 1, MultipleLines cannot be 'No phone service'."
            )

        # Internet consistency
        internet_fields = [
            self.OnlineSecurity,
            self.OnlineBackup,
            self.DeviceProtection,
            self.TechSupport,
            self.StreamingTV,
            self.StreamingMovies,
        ]

        if self.HasInternet == 0:
            if self.InternetService != "No internet service":
                raise ValueError(
                    "If HasInternet is 0, InternetService must be "
                    "'No internet service'."
                )

            if any(
                value != "No internet service"
                for value in internet_fields
            ):
                raise ValueError(
                    "If HasInternet is 0, all internet add-on fields must be "
                    "'No internet service'."
                )

        return self
