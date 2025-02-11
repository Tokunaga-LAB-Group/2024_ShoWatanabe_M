ROOT_DIR		=	$(shell pwd)
CSV_DIR			=	$(ROOT_DIR)/csv
DATA_DIR		=	$(ROOT_DIR)/data
ANNOTATION_PER	=	0.01
ANNOTATION_DIR	=	$(CSV_DIR)/$(TARGET_NAMES)/$(MODEL_TYPE)/$(ANNOTATION_TYPE)/$(ANNOTATION_POINT)/$(NUM)
ANNOTATION_CSV	=	$(ANNOTATION_DIR)/$(ANNOTATION_PER).csv
ANNOTATION_TYPE	=	random
ANNOTATION_POINT=	random
USE_SAMPLING	=   True

DATASET_ROOT_DIR=	$(ROOT_DIR)/data
DATASET_DIR		=	$(ROOT_DIR)/data/$(TARGET_ID)

RESULT_DIR		=	$(ROOT_DIR)/res/$(TARGET_NAMES)
C_RESULT_DIR	=	$(RESULT_DIR)/cospa-pointnet/$(ANNOTATION_TYPE)/$(ANNOTATION_POINT)/$(NUM)/L-$(LR)_A-$(ANNOTATION_PER)
CPP_RESULT_DIR	=	$(RESULT_DIR)/cospa-pointnetpp/$(ANNOTATION_TYPE)/$(ANNOTATION_POINT)/$(NUM)/L-$(LR)_A-$(ANNOTATION_PER)
C_MODEL_PATH	=	$(C_RESULT_DIR)/model.pth
CPP_MODEL_PATH	=	$(CPP_RESULT_DIR)/model.pth
CHECK_POINT_EPOCH =
# C_CHECK_POINT_MODEL_PATH = $(C_RESULT_DIR)/log/model_$(CHECK_POINT_EPOCH).pth
MODEL_TYPE		=	pointnet
P_RESULT_DIR	=	$(RESULT_DIR)/$(MODEL_TYPE)/$(NUM)/L-$(P_LR)
P_MODEL_PATH	=	$(P_RESULT_DIR)/model.pth
# P_CHECK_POINT_MODEL_PATH = $(P_RESULT_DIR)/log/model_$(CHECK_POINT_EPOCH).pth

NUM				=	1

# Train Settings
EPOCH			=	200
BATCH_SIZE		=	8
LR				=	1e-4
TRAIN_SPLIT		=	0.5


P_LR			=	1e-4

TARGET_NAMES	=	Cap

GPU				=	0

all: annotation cospa test

annotation: $(ANNOTATION_CSV)


annotation_multi: $(ANNOTATION_CSV)
$(ANNOTATION_CSV):
# annotation_multi:
	mkdir -p $(ANNOTATION_DIR)
	python3 $(ROOT_DIR)/annotation_multi.py \
		--annotation_csv $(ANNOTATION_CSV) \
		--annotation_per $(ANNOTATION_PER) \
		--target_names $(TARGET_NAMES) \
		--train_split $(TRAIN_SPLIT) \
		--dataset_dir $(DATASET_ROOT_DIR) \
		--use_sampling $(USE_SAMPLING) \
		--annotation_type $(ANNOTATION_TYPE) \
		--annotation_point $(ANNOTATION_POINT)

annotation_multi2:
# annotation_multi:
	mkdir -p $(ANNOTATION_DIR)
	python3 $(ROOT_DIR)/annotation_multi_const2.py \
		--annotation_csv $(ANNOTATION_CSV) \
		--annotation_per $(ANNOTATION_PER) \
		--target_names $(TARGET_NAMES) \
		--train_split $(TRAIN_SPLIT) \
		--dataset_dir $(DATASET_ROOT_DIR) \
		--use_sampling $(USE_SAMPLING) \
		--annotation_type $(ANNOTATION_TYPE) \
		--annotation_point $(ANNOTATION_POINT)

re_annotation:
	rm -rf $(ANNOTATION_CSV)
	make annotation

cospa: $(C_MODEL_PATH)
$(C_MODEL_PATH):
	mkdir -p $(C_RESULT_DIR)
	python3 $(ROOT_DIR)/train.py \
		--annotation_csv $(ANNOTATION_CSV) \
		--dataset_dir $(DATASET_ROOT_DIR) \
		--target_names $(TARGET_NAMES) \
		--res_dir $(C_RESULT_DIR) \
		--epoch $(EPOCH) \
		--batch_size $(BATCH_SIZE) \
		--gpu $(GPU) \
		--lr $(LR) \
		--train_split $(TRAIN_SPLIT) \
		--cospa_train True | tee $(C_RESULT_DIR)/train.txt

cospapp: $(CPP_MODEL_PATH)
$(CPP_MODEL_PATH):
	mkdir -p $(CPP_RESULT_DIR)
	python3 $(ROOT_DIR)/cospapp.py \
		--annotation_csv $(ANNOTATION_CSV) \
		--dataset_dir $(DATASET_ROOT_DIR) \
		--target_names $(TARGET_NAMES) \
		--res_dir $(CPP_RESULT_DIR) \
		--epoch $(EPOCH) \
		--batch_size $(BATCH_SIZE) \
		--gpu $(GPU) \
		--lr $(LR) \
		--train_split $(TRAIN_SPLIT) | tee $(CPP_RESULT_DIR)/train.txt

c_show:
	tensorboard --logdir=$(C_RESULT_DIR)/log

c_test: $(C_RESULT_DIR)/test.txt
$(C_RESULT_DIR)/test.txt:
	python3 $(ROOT_DIR)/test.py \
		--dataset_dir $(DATASET_ROOT_DIR) \
		--res_dir $(C_RESULT_DIR) \
		--target_names $(TARGET_NAMES) \
		--model_path $(C_MODEL_PATH) \
		--gpu $(GPU) \
		--train_split $(TRAIN_SPLIT) \
		--batch_size $(BATCH_SIZE) | tee $(C_RESULT_DIR)/test.txt

c_test2: $(CPP_RESULT_DIR)/test.txt
$(CPP_RESULT_DIR)/test.txt:
	python3 $(ROOT_DIR)/test2.py \
		--dataset_dir $(DATASET_ROOT_DIR) \
		--res_dir $(CPP_RESULT_DIR) \
		--target_names $(TARGET_NAMES) \
		--model_path $(CPP_MODEL_PATH) \
		--gpu $(GPU) \
		--train_split $(TRAIN_SPLIT) \
		--batch_size $(BATCH_SIZE) | tee $(CPP_RESULT_DIR)/test.txt

pointnet:
	mkdir -p $(P_RESULT_DIR)
	python3 $(ROOT_DIR)/train.py \
		--annotation_csv $(ANNOTATION_CSV) \
		--dataset_dir $(DATASET_ROOT_DIR) \
		--target_names $(TARGET_NAMES) \
		--res_dir $(P_RESULT_DIR) \
		--epoch $(EPOCH) \
		--gpu $(GPU) \
		--batch_size $(BATCH_SIZE) \
		--train_split $(TRAIN_SPLIT) \
		--lr $(P_LR) | tee $(P_RESULT_DIR)/train.txt

pointnet2:
	mkdir -p $(P_RESULT_DIR)
	python3 $(ROOT_DIR)/pointnetp2.py \
		--annotation_csv /data/Users/watanabe/M2/GlobalPointCoSPA/csv/simple/Mug/0/0.01.csv \
		--dataset_dir $(DATASET_ROOT_DIR) \
		--target_names $(TARGET_NAMES) \
		--res_dir $(P_RESULT_DIR) \
		--epoch $(EPOCH) \
		--gpu $(GPU) \
		--batch_size $(BATCH_SIZE) \
		--lr $(P_LR) | tee $(P_RESULT_DIR)/train.txt

p_test:
	python3 $(ROOT_DIR)/test.py \
		--dataset_dir $(DATASET_ROOT_DIR) \
		--res_dir $(P_RESULT_DIR) \
		--target_names $(TARGET_NAMES) \
		--model_path $(P_MODEL_PATH) \
		--gpu $(GPU) \
		--train_split $(TRAIN_SPLIT) \
		--batch_size $(BATCH_SIZE) | tee $(P_RESULT_DIR)/test.txt

p_test2:
	python3 $(ROOT_DIR)/test2.py \
		--dataset_dir $(DATASET_ROOT_DIR) \
		--res_dir $(P_RESULT_DIR) \
		--target_names $(TARGET_NAMES) \
		--model_path $(P_MODEL_PATH) \
		--gpu $(GPU) \
		--train_split $(TRAIN_SPLIT) \
		--batch_size $(BATCH_SIZE) | tee $(P_RESULT_DIR)/test.txt

p_show:
	tensorboard --logdir=$(P_RESULT_DIR)/log
