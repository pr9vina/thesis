def validate_argument(value, allowed_values, argument_name):
    if value not in allowed_values:
        raise ValueError(f"Invalid value '{value}' for {argument_name}. Allowed values: {allowed_values}")
