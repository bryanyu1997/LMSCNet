import importlib

def get_model(_cfg, dataset):

  nbr_classes = dataset.nbr_classes
  grid_dimensions = dataset.grid_dimensions
  class_frequencies = dataset.class_frequencies
  selected_model = _cfg._dict['MODEL']['TYPE']
  if 'KWARGS' in _cfg._dict['MODEL']:
    kwargs_dict = _cfg._dict['MODEL']['KWARGS']
  else:
    kwargs_dict = {}
  ModelFile = importlib.import_module(_cfg._dict['MODEL']['FILE'])
  model = getattr(ModelFile,selected_model)(class_num=nbr_classes, input_dimensions=grid_dimensions, class_frequencies=class_frequencies, **kwargs_dict)

  return model