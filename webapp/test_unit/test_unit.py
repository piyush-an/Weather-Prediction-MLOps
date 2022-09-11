import predict


def test_prepare_features():
    user_input = {
        "HourlyDryBulbTemperature": 76,
        "LastHourDryBulbTemperature": 82,
        "HourlyPrecipitation": 0,
        "HourlyWindSpeed": 8,
    }
    status, func_output_df = predict.prepare_features(user_input)
    expected_output = {
        "LastHourDryBulbTemperature": 82,
        "HourlyPrecipitation": 0,
        "HourlyWindSpeed": 8,
    }
    func_output = func_output_df.to_dict(orient='records')[0]
    assert status == True
    assert expected_output == func_output


def test_prepare_features_invalid_input():
    user_input = {
        "LastHourDryBulbTemperature": 82,
        "HourlyPrecipitation": 0,
        "HourlyWindSpeed": 8,
    }
    status = predict.prepare_features(user_input)
    assert status == False
