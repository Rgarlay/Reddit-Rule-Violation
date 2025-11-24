from dataclasses import dataclass

@dataclass

class DataIngestionArtifact:
    train_file_path: str
    test_file_path: str

@dataclass

class DataValidationArtifact:
    valid_train_file_path: str
    valid_test_file_path: str
    invalid_train_file_path: str
    invalid_test_file_path: str


@dataclass
class DataTransformationArtifact:
    train_obj_file_path: str
    test_obj_file_path: str

@dataclass
class ModelTrainerArtifact:
    train_artifact_metric: float
    test_artifact_metric: float
    train_model_artifact: str
    