LINK_PARA = checkpoint logdir result weights
IMG_PATH = ./data/imgs
link:
	bash build.sh
clean:
	rm -rf ${LINK_PARA}
move_imgs:
	python data_fire.py move_imgs --target_root ${IMG_PATH}
build_file_root:
	python data_fire.py build_data --root ./data --dir train_dir --det Detections
build_index:
	python data_fire.py build_idx
build_all:
	make move_imgs build_index
label:
	./tool/labelImg
train:
	python train.py
CUDA:
	bash make.sh
trainFSSD:
	python train.py --models FSSD
trainRefineDet:
	python refinedet_train_test.py train
TestRefineDet:
	python refinedet_train_test.py --fire test --resume_net True

