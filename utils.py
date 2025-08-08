import hashlib

SENSITIVE_FIELDS = ["name", "ip_address"]

def hash_field(data: str) -> str:
    return hashlib.sha256(data.encode()).hexdigest()

def anonymize_data(data: dict) -> dict:
    for field in SENSITIVE_FIELDS:
        if field in data and data[field]:
            data[field] = hash_field(data[field])
    return data
