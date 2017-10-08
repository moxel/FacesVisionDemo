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
print(mean)


age_net_pretrained=folder + '/age_net.caffemodel'
age_net_model_file=folder + '/deploy_age.prototxt'
age_net = caffe.Classifier(age_net_model_file, age_net_pretrained,
                       mean=mean,
                       channel_swap=(2,1,0),
                       raw_scale=255,
                       image_dims=(256, 256))

age_list=['(0, 2)','(4, 6)','(8, 12)','(15, 20)','(25, 32)','(38, 43)','(48, 53)','(60, 100)']

def predict(image):
    image = image.to_numpy_rgb()[:, :, :3]
    image = np.array(image, dtype='float32')
    pred = age_net.predict([image])[0]
    result = [(age_list[i], float(pred[i])) for i in range(len(pred))]
    result = sorted(result, key=lambda x: x[1], reverse=True)
    return {
        'age': result
    }


if __name__ == '__main__':
    image = moxel.space.Image.from_file('cnn_age_gender_models_and_data.0.0.2/example_image.jpg')
    predict(image)




