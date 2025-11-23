import json

class InvalidJudgeResponse(Exception):
    pass

def validate_judge_response(data: dict, EXPECTED_FIELDS: dict) -> dict:
    # Check all expected fields exist
    for field, field_type in EXPECTED_FIELDS.items():
        if field not in data:
            raise InvalidJudgeResponse(f"Missing required field: {field}")
        if not isinstance(data[field], field_type):
            raise InvalidJudgeResponse(
                f"Field '{field}' has invalid type: expected {field_type}, got {type(data[field])}"
            )
    
    # Extra checks for list contents
    if not all(isinstance(x, str) for x in data["categories"]):
        raise InvalidJudgeResponse("All items in 'categories' must be strings.")
    
    return data
