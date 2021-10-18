from . import modelling
import json
model = modelling.load_saved_model( '/home/theo/sandslash/checkpoints2/model2_3.h5')
model, mask = modelling.create_pretrained_pruned_model(model)
model.save('/home/theo/armadillin/git-arm/src/armadillin/trained_model/model_small.h5')
json.dump(mask.tolist(), open('/home/theo/armadillin/git-arm/src/armadillin/trained_model/mask_small.json', 'wt'))
