# Run the server from here.
from app import app
from app.model.controllers import *
import load_models
from face_mask_detector import detect_mask_video


faceNet, maskNet=load_models.load()
detect_mask_video.init_video_streaming(faceNet, maskNet)






if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
