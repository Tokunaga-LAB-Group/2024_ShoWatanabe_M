function exec()
{
	echo ======== No.$3 $1 $2 =======
	# make annotation_multi_$6 LR=$1 ANNOTATION_PER=$2 GPU=$4 TARGET_NAMES=$5 NUM=$3 ANNOTATION_TYPE=$6 EPOCH=100
    # make cospa LR=$1 ANNOTATION_PER=$2 GPU=$4 TARGET_NAMES=$5 NUM=$3 ANNOTATION_TYPE=$6 EPOCH=100
	# make c_test LR=$1 ANNOTATION_PER=$2 GPU=$4 TARGET_NAMES=$5 NUM=$3 ANNOTATION_TYPE=$6 EPOCH=100
}

function exec-pointnet()
{
	echo ======== No.$3 $1 $2 =======
	# make annotation_multi LR=$1 ANNOTATION_PER=$2 GPU=$4 TARGET_NAMES=$5 NUM=$3
    make pointnet2 P_LR=$1 ANNOTATION_PER=$2 GPU=$4 TARGET_NAMES=$5 NUM=$3 EPOCH=100 MODEL_TYPE=pointnetpp
	make p_test2 P_LR=$1 ANNOTATION_PER=$2 GPU=$4 TARGET_NAMES=$5 NUM=$3 EPOCH=100 MODEL_TYPE=pointnetpp
}
for cate in Mug Airplane
do
	for num in 0 1 2 3 4
	do
		for ap in 1
		do
			for lr in 1e-4
			do
				for type in const
				do
					if [ $ap -eq 1 ]; then
						exec-pointnet $lr $ap $num 1 $cate $type
					else
						exec $lr $ap $num 1 $cate $type
					fi
				done
			done
		done
	done
done
