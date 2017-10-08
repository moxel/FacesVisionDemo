import caffe
import caffe.io
import numpy as np
import moxel

folder = 'cnn_emotion'

emotion_net_pretrained=folder + '/EmotiW_VGG_S.caffemodel'
emotion_net_model_file=folder + '/deploy.prototxt'
emotion_net = caffe.Classifier(emotion_net_model_file, emotion_net_pretrained)

with open('emotions.txt', 'r') as f:
    lines = f.readlines()
    labels = [line.replace('\n', '') for line in lines]

def predict(image):
    image = image.to_numpy_rgb()[:, :, :3]
    image = np.array(image, dtype='float32')
    pred = emotion_net.predict([image])[0]
    result = [(labels[i], float(pred[i])) for i in range(len(pred))]
    return {
        'emotion': result
    }


if __name__ == '__main__':
    image = moxel.space.Image.from_file('cnn_age_emotion_models_and_data.0.0.2/example_image.jpg')
    predict(image)




