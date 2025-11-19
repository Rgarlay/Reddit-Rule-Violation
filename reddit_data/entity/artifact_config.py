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
    transformed_obj_file_path: str
    train_obj_file_path: str
    test_obj_file_path: str