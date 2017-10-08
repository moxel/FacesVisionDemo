import moxel

model = moxel.Model('moxel/awesome:latest', where='localhost')
image = moxel.space.Image.from_file('cnn_age_gender_models_and_data.0.0.2/example_image.jpg')
output = model.predict(image=image)
print(output['age'].to_object())
