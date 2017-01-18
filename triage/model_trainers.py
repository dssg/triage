class ModelTrainer(object):
    def __init__(self, training_set_path, test_set_path, model_config, trained_model_path):
        self.training_set_path = training_set_path
        self.test_set_path = test_set_path
        self.model_config = model_config
        self.trained_model_path = trained_model_path

    def train(self):
        """TODO:
        retrieve x_train, x_test, y_train, y_test from s3 path
        for each model from model config:
            instantiate model
            run
            save pickle to s3 (trained_model_path + '/' + uuid.uuid4() or something)
            return id for each model
        """
        return ['uuid1', 'uuid2']
