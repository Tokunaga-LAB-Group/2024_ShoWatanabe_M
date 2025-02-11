function exec()
{
	echo ======== No.$3 $1 $2 $6 $7 =======
	make annotation_multi LR=$1 ANNOTATION_PER=$2 GPU=$4 TARGET_NAMES=$5 NUM=$3 ANNOTATION_TYPE=$6 EPOCH=100 ANNOTATION_POINT=$7 MODEL_TYPE=pointnetpp USE_SAMPLING=True TRAIN_SPLIT=0.5
    make cospapp LR=$1 ANNOTATION_PER=$2 GPU=$4 TARGET_NAMES=$5 NUM=$3 ANNOTATION_TYPE=$6 EPOCH=100 ANNOTATION_POINT=$7 USE_SAMPLING=True MODEL_TYPE=pointnetpp TRAIN_SPLIT=0.5
	make c_test2 LR=$1 ANNOTATION_PER=$2 GPU=$4 TARGET_NAMES=$5 NUM=$3 ANNOTATION_TYPE=$6 EPOCH=100 ANNOTATION_POINT=$7 USE_SAMPLING=True MODEL_TYPE=pointnetpp TRAIN_SPLIT=0.5
}

function exec-pointnet()
{
	echo ======== No.$3 $1 $2 =======
	# make annotation_multi LR=$1 ANNOTATION_PER=$2 GPU=$4 TARGET_NAMES=$5 NUM=$3
    # make pointnet2 P_LR=$1 ANNOTATION_PER=$2 GPU=$4 TARGET_NAMES=$5 NUM=$3 EPOCH=100 MODEL_TYPE=pointnetpp
	# make p_test2 P_LR=$1 ANNOTATION_PER=$2 GPU=$4 TARGET_NAMES=$5 NUM=$3 EPOCH=100 MODEL_TYPE=pointnetpp
}
for cate in Mug Airplane
do
	for num in 2
	do
		for ap in 0.01 0.1 0.5
		do
			for point in random uniformed boundary interior
			do
				for type in simple const const2 rev random rev2
				do
					if [ $ap -eq 1 ]; then
						exec-pointnet 1e-4 $ap $num 3 $cate $type
					else
						exec 1e-4 $ap $num 2 $cate $type $point
					fi
				done
			done
		done
	done
done
