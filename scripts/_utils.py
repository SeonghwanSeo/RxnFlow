def parse_temperature(temperature_param: str) -> tuple[str, list[float]]:
    temperature_info = temperature_param.split("-")
    sample_dist = temperature_info[0]
    dist_params = list(map(float, temperature_info[1:]))
    assert sample_dist in ("constant", "uniform", "loguniform", "gamma", "beta")
    if sample_dist == "constant":
        assert len(dist_params) == 1, "constant temperature requires only one parameter; e.g. constant-32"
    else:
        assert len(dist_params) == 2, f"{sample_dist} temperature requires two parameters."
    return sample_dist, dist_params
