from reddit_data.logging.logger import logging
from reddit_data.exception.exception import CustomException
import os,sys
from reddit_data.utils.main_utils.utils import  save_pickle_file, tokenization_of_text
from reddit_data.utils.main_utils.utils import clean_text, get_vocab_size, save_txt_file

import pandas as pd
from reddit_data.entity.artifact_config import DataValidationArtifact, DataTransformationArtifact
from reddit_data.entity.entity_config import DataTransformationConfig
import sentencepiece as spm



class DataTransformation:
    def __init__(self, data_validation_artifact:DataValidationArtifact, data_transformation_config:DataTransformationConfig):
        try:
            logging.info("Initializing DataTransformation")
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config
        except Exception as e:
            raise CustomException(e,sys)
    
    def taking_apart_and_concatinating(self, df:pd.DataFrame):
        try:
            logging.info("Starting data concatenation")
            x_positive_1_df = df[['rule', 'positive_example_1']].rename(columns={'positive_example_1': 'body'})
            x_positive_2_df = df[['rule', 'positive_example_2']].rename(columns={'positive_example_2': 'body'})
            x_negative_1_df = df[['rule', 'negative_example_1']].rename(columns={'negative_example_1': 'body'})
            x_negative_2_df = df[['rule', 'negative_example_2']].rename(columns={'negative_example_2': 'body'})
            x_body_df = df[['rule','body','rule_violation']]

            x_positive_1_df['rule_violation'] = 1
            x_positive_2_df['rule_violation'] = 1
            x_negative_1_df['rule_violation'] = 0
            x_negative_2_df['rule_violation'] = 0

            df_concatinated = pd.concat([x_positive_1_df, x_positive_2_df, 
                                         x_negative_1_df, x_negative_2_df, 
                                         x_body_df], axis = 0).reset_index(drop=True)

            logging.info("Data concatenation completed")
            return df_concatinated

        except Exception as e:
            raise CustomException(e,sys)
        
    def training_tokenizer(self, text_file_path, output_dir_name, vocab_size):        ##Write these 3 into constants
        '''
        We train and define the tokenizer here.
        Output_train_dir will have .vocab and .model obj. 
        To specify it, just write dir_name/tokenizer.
        Both objs will autometically save there.
        Get vocab size from utils
        '''
        try:
            
            spm.SentencePieceTrainer.train(    
                input = text_file_path,
                model_prefix = output_dir_name,
                vocab_size = vocab_size,
                character_coverage = 1,
                model_type = 'bpe',
                control_symbols=["<pad>", "<sos>", "<eos>"],
                hard_vocab_limit=False,
                byte_fallback = True
            )

            tokenizer_obj = spm.SentencePieceProcessor()
            tokenizer_obj.load(f"{output_dir_name}.model")
            pad_id = tokenizer_obj.piece_to_id("<pad>")#same in both lang.

            return tokenizer_obj, pad_id
        
        except Exception as e:
            raise CustomException(e,sys)
    
            
    def initiate_data_transformation(self):
        try:

            logging.info("Starting data transformation process")

            train_file_path = self.data_validation_artifact.valid_train_file_path
            test_file_path = self.data_validation_artifact.valid_test_file_path

            logging.info("Reading train and test CSV files")
            train_df = pd.read_csv(train_file_path)
            test_df = pd.read_csv(test_file_path)

            train_df = train_df.head(10)
            test_df = test_df.head(5)

            logging.info("Concatenating train data")
            train_df_concatinated = self.taking_apart_and_concatinating(train_df)
            logging.info("Concatenating test data")
            test_df_concatinated = self.taking_apart_and_concatinating(test_df)

            ## Body Text
            body_text = 'body'
            
            logging.info("Cleaning train body text")
            train_body_text_file = list(train_df_concatinated[body_text])
            logging.info("Cleaning test body text")
            test_body_text_file = list(test_df_concatinated[body_text])

            train_body_text_cleaned = clean_text(train_body_text_file)
            test_body_text_cleaned = clean_text(test_body_text_file)

            body_vocab_size = get_vocab_size(train_body_text_file)

            logging.info('Saving body text file for tokenization further.') 
            
            body_text_save_path = self.data_transformation_config.data_transformation_body_cleaned_file_path

            os.makedirs(os.path.dirname(os.path.join(self.data_transformation_config.data_transformation_body_cleaned_file_path)), exist_ok=True)

            ### This is for trial run. Comment and subsitute actual full vocab_sizes for full scale model.
            vocab_size = 334
            save_txt_file(train_body_text_cleaned,
                          body_text_save_path, replace=False)
            
            body_tokenizer, pad_id = self.training_tokenizer(text_file_path = body_text_save_path,
                                    output_dir_name=self.data_transformation_config.data_transformation_body_artifact_save,
                                    vocab_size=334)
            
            body_training_text_tokenized = tokenization_of_text(tokenizer=body_tokenizer,
                                                                     file_name=train_body_text_cleaned,
                                                                     pad_id=pad_id)
            
            body_testing_text_tokenized = tokenization_of_text(tokenizer=body_tokenizer,
                                                                    file_name=test_body_text_cleaned,
                                                                    pad_id=pad_id)
           

            ##Rule part
            rule_column = 'rule'

            train_rule_file = list(train_df_concatinated[rule_column])
            test_rule_file = list(test_df_concatinated[rule_column])

            rule_vocab_size = get_vocab_size(train_rule_file)

            rule_text_save_path = self.data_transformation_config.data_transformation_rule_cleaned_file_path

            save_txt_file(train_rule_file,
                          rule_text_save_path, replace=False)
            
            rule_tokenizer, _ = self.training_tokenizer(text_file_path = rule_text_save_path,
                                    output_dir_name=self.data_transformation_config.data_transformation_rule_artifact_save,
                                    vocab_size=334)
            
            rule_training_text_tokenized= tokenization_of_text(tokenizer=rule_tokenizer,
                                                                     file_name=train_rule_file,
                                                                     pad_id=pad_id)
            
            rule_testing_text_tokenized = tokenization_of_text(tokenizer=rule_tokenizer,
                                                                    file_name=test_rule_file,
                                                                    pad_id=pad_id)

            train_df_concatinated[body_text] = body_training_text_tokenized
            test_df_concatinated[body_text] = body_testing_text_tokenized

            train_df_concatinated[rule_column] = rule_training_text_tokenized
            test_df_concatinated[rule_column] = rule_testing_text_tokenized

            dirname = os.path.dirname(self.data_transformation_config.data_transformed_train_file_path)

            logging.info("Creating directories if missing")
            os.makedirs(dirname, exist_ok=True)

            logging.info("Saving transformed train and test pickle files")
            save_pickle_file(file_to_save=train_df_concatinated,
                            file_path = self.data_transformation_config.data_transformed_train_file_path)
            
            save_pickle_file(file_to_save = test_df_concatinated,
                             file_path=self.data_transformation_config.data_transformed_test_file_path)
            
            logging.info("Creating final DataTransformationArtifact")
            data_transformation_artifact = DataTransformationArtifact(
                train_obj_file_path = self.data_transformation_config.data_transformed_train_file_path,
                test_obj_file_path = self.data_transformation_config.data_transformed_test_file_path,
                pad_id_token=pad_id
            )
            logging.info("Data transformation completed successfully")
            return data_transformation_artifact
        except Exception as e:
            raise CustomException(e,sys)