from firebase_auth import get_db


def save_prediction(user_id: str, data: dict):
    db = get_db()
    doc_ref = db.collection("users").document(user_id).collection("predictions").document()
    data["user_id"] = user_id
    doc_ref.set(data)
    return doc_ref.id


def get_user_predictions(user_id: str, limit: int = 50):
    db = get_db()
    docs = (
        db.collection("users")
        .document(user_id)
        .collection("predictions")
        .order_by("timestamp", direction="DESCENDING")
        .limit(limit)
        .stream()
    )
    return [{"id": d.id, **d.to_dict()} for d in docs]
