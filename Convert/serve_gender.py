import caffe
import caffe.io
import numpy as np
import moxel

folder = 'cnn_age_gender_models_and_data.0.0.2'

mean_filename=folder + '/mean.binaryproto'
proto_data = open(mean_filename, "rb").read()
a = caffe.io.caffe_pb2.BlobProto.FromString(proto_data)
mean  = caffe.io.blobproto_to_array(a)[0]
mean = caffe.io.resize_image(np.transpose(mean, (1, 2, 0)), (227, 227))
mean = np.transpose(mean, (2, 0, 1))


gender_net_pretrained=folder + '/gender_net.caffemodel'
gender_net_model_file=folder + '/deploy_gender.prototxt'
gender_net = caffe.Classifier(gender_net_model_file, gender_net_pretrained,
                       mean=mean,
                       channel_swap=(2,1,0),
                       raw_scale=255,
                       image_dims=(256, 256))

with open('genders.txt', 'r') as f:
    lines = f.readlines()
    labels = [line.replace('\n', '') for line in lines]

def predict(image):
    image = image.to_numpy_rgb()[:, :, :3]
    image = np.array(image, dtype='float32')
    pred = gender_net.predict([image])[0]
    result = [(labels[i], float(pred[i])) for i in range(len(pred))]
    return {
        'gender': result
    }


if __name__ == '__main__':
    image = moxel.space.Image.from_file('cnn_age_gender_models_and_data.0.0.2/example_image.jpg')
    predict(image)




