from io import BytesIO
import pickle


def deserialize_model(payload: bytes):
    return pickle.load(BytesIO(payload))
