from dashboard.app import build_derived_features


def test_derived_features_without_internet():
    result = build_derived_features(
        internet_service="No internet service",
        online_security="No internet service",
        online_backup="No internet service",
        device_protection="No internet service",
        tech_support="No internet service",
        streaming_tv="No internet service",
        streaming_movies="No internet service",
        has_phone=1,
        payment_method="Mailed check",
    )

    assert result["HasInternet"] == 0
    assert result["HasPhone"] == 1
    assert result["InternetAddOnCount"] == 0
    assert result["StreamingCount"] == 0
    assert result["HasTechProtection"] == 0
    assert result["AutomaticPayment"] == 0


def test_derived_features_with_full_services():
    result = build_derived_features(
        internet_service="Fiber optic",
        online_security="Yes",
        online_backup="Yes",
        device_protection="Yes",
        tech_support="Yes",
        streaming_tv="Yes",
        streaming_movies="Yes",
        has_phone=1,
        payment_method="Credit card (automatic)",
    )

    assert result["HasInternet"] == 1
    assert result["HasPhone"] == 1
    assert result["InternetAddOnCount"] == 4
    assert result["StreamingCount"] == 2
    assert result["HasTechProtection"] == 1
    assert result["AutomaticPayment"] == 1


def test_automatic_payment_bank_transfer():
    result = build_derived_features(
        internet_service="DSL",
        online_security="No",
        online_backup="No",
        device_protection="No",
        tech_support="No",
        streaming_tv="No",
        streaming_movies="No",
        has_phone=1,
        payment_method="Bank transfer (automatic)",
    )

    assert result["AutomaticPayment"] == 1


def test_manual_payment_not_automatic():
    result = build_derived_features(
        internet_service="DSL",
        online_security="No",
        online_backup="No",
        device_protection="No",
        tech_support="No",
        streaming_tv="No",
        streaming_movies="No",
        has_phone=1,
        payment_method="Electronic check",
    )

    assert result["AutomaticPayment"] == 0


def test_tech_protection_from_tech_support():
    result = build_derived_features(
        internet_service="DSL",
        online_security="No",
        online_backup="No",
        device_protection="No",
        tech_support="Yes",
        streaming_tv="No",
        streaming_movies="No",
        has_phone=1,
        payment_method="Mailed check",
    )

    assert result["HasTechProtection"] == 1


def test_tech_protection_from_device_protection():
    result = build_derived_features(
        internet_service="DSL",
        online_security="No",
        online_backup="No",
        device_protection="Yes",
        tech_support="No",
        streaming_tv="No",
        streaming_movies="No",
        has_phone=1,
        payment_method="Mailed check",
    )

    assert result["HasTechProtection"] == 1
