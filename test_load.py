from tensorflow.keras.models import load_model
from tkan import TKAN

# hack để ignore quantization_config
def custom_objects_fix():
    from keras.layers import Dense

    class DenseFix(Dense):
        def __init__(self, *args, **kwargs):
            kwargs.pop("quantization_config", None)  # bỏ field lỗi
            super().__init__(*args, **kwargs)

    return {"TKAN": TKAN, "Dense": DenseFix}


model = load_model(
    "model/mtkan_model.keras",
    custom_objects=custom_objects_fix(),
    compile=False
)

print("Load OK")
model.summary()
