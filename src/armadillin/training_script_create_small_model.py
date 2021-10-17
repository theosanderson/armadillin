import modelling
import json
model = modelling.load_saved_model( './trained_model/model.h5')
model, mask = modelling.create_pretrained_pruned_model(model)
model.save('./trained_model/model_small.h5')
json.dump(mask.tolist(), open('./trained_model/mask_small.json', 'wt'))
